from absl.app import run
import math
from tqdm import tqdm
from PIL import Image
import decord
from functools import cached_property
import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from transformers import GenerationConfig
from tux import (
    define_flags_with_default, StreamingCheckpointer, JaxDistributedConfig,
    set_random_seed, get_float_dtype_by_name, JaxRNG, next_rng,
    match_partition_rules, make_shard_and_gather_fns,
    with_sharding_constraint, tree_apply, open_file
)
from lwm.vision_llama import VideoLLaMAConfig, FlaxVideoLLaMAForCausalLM
from lwm.vqgan import VQGAN

import argparse
import gradio as gr
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description='LWM Demo')
    parser.add_argument('--llama_tokenizer_path', default='/root/jinyfeng/models/LWM/LWM-Chat-1M-Jax/tokenizer.model', help='tokenizer.model')
    parser.add_argument('--vqgan_checkpoint', default='/root/jinyfeng/models/LWM/LWM-Chat-1M-Jax/vqgan', help='vqgan')
    parser.add_argument('--lwm_checkpoint', default='/root/jinyfeng/models/LWM/LWM-Chat-1M-Jax/params', help='params')
    parser.add_argument('--mesh_dim', default='!1,1,-1,1', help='mesh_dim')
    parser.add_argument('--dtype', default='fp32', help='dtype')
    parser.add_argument('--load_llama_config', default='7b', help='load_llama_config')
    parser.add_argument('--max_n_frames', default=8, help='max_n_frames')
    parser.add_argument('--update_llama_config', default="dict(sample_mode='text',theta=50000000,max_sequence_length=131072,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,remat_attention='',scan_mlp=False,scan_mlp_chunk_size=2048,remat_mlp='',remat_block='',scan_layers=True)", help='update_llama_config')
    
    parser.add_argument(
        '--server_port', type=int, default=8080, help='the gradio server port')
    args = parser.parse_args()
    return args

FLAGS, FLAGS_DEF = define_flags_with_default(
    prompt="",
    input_file="",
    vqgan_checkpoint="",
    temperature=0.2,
    max_n_frames=8,
    seed=1234,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    tokenizer=VideoLLaMAConfig.get_tokenizer_config(),
    llama=VideoLLaMAConfig.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
) 

class Sampler:
    def __init__(self):
        self.mesh = VideoLLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
        self.vqgan = VQGAN(FLAGS.vqgan_checkpoint, replicate=False)
        self.prefix_tokenizer = VideoLLaMAConfig.get_tokenizer(
            FLAGS.tokenizer, truncation_side='left', padding_side='left'
        )
        self.tokenizer = VideoLLaMAConfig.get_tokenizer(FLAGS.tokenizer)
        self.n_tokens_per_frame = 257
        self.min_buffer_size = 256
        self.sharded_rng = next_rng()
        self._load_model()

    @property
    def block_size(self):
        return max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size) * self.mesh.shape['sp']
    
    @property
    def data_dim(self):
        return self.mesh.shape['dp'] * self.mesh.shape['fsdp']

    def _process_frame(self, image, size):
        width, height = image.size
        if width < height:
            new_width = size
            new_height = int(size * height / width)
        else:
            new_height = size
            new_width = int(size * width / height)
        image = image.resize((new_width, new_height))

        left = (new_width - size) / 2
        top = (new_height - size) / 2
        right = (new_width + size) / 2
        bottom = (new_height + size) / 2
        image = image.crop((left, top, right, bottom))
        return np.array(image, dtype=np.float32) / 127.5 - 1
    
    def _read_process_vision(self, path, max_n_frames):
        f = open_file(path, 'rb')
        if path.endswith('.png') or path.endswith('.jpg'):
            image = Image.open(f).convert('RGB')
            vision = self._process_frame(image, 256)[None]
        else:
            vr = decord.VideoReader(f, ctx=decord.cpu(0))
            duration = len(vr)
            if duration <= max_n_frames:
                frame_id_list = list(range(duration))
            else:
                frame_id_list = np.linspace(0, duration - 1, max_n_frames, dtype=int).tolist()
            video = vr.get_batch(frame_id_list).asnumpy()
            vision = np.stack([self._process_frame(Image.fromarray(frame), 256) for frame in video])

        B = 1
        encodings = []
        for i in range(0, len(vision), 1):
            v = vision[i:i + B]
            if len(v) % B == 0:
                n_pad = 0
            else:
                n_pad = B - len(v) % B
            v = np.pad(v, ((n_pad, 0), (0, 0), (0, 0), (0, 0)))
            enc = jax.device_get(self.vqgan.encode(v))[1].astype(int)
            enc = enc[n_pad:]
            for t in range(len(enc)):
                encodings.extend(enc[t].reshape(-1).tolist())
                if t == len(enc) - 1:
                    encodings.append(8193)
                else:
                    encodings.append(8192)
        return encodings

    def construct_input(self, prompts, max_n_frames):
        max_input_length = max_n_frames * self.n_tokens_per_frame + self.min_buffer_size
        max_input_length = int(math.ceil(max_input_length / self.block_size) * self.block_size)

        vision_start = self.tokenizer.encode('<vision>')
        vision_end = self.tokenizer.encode('</vision>')

        input_ids = np.zeros((len(prompts), max_input_length), dtype=int)
        vision_masks = np.zeros((len(prompts), max_input_length), dtype=bool)
        attention_mask = np.zeros((len(prompts), max_input_length), dtype=int)
        for i, prompt in enumerate(tqdm(prompts)):
            vision = self._read_process_vision(prompt['input_path'], max_n_frames)
            text_1 = self.tokenizer.encode(f"<s>You are a helpful assistant. USER: {prompt['question']}\n")
            tail = self.tokenizer.encode(" ASSISTANT:")
            
            tokens, vm = [], []
            tokens.extend(text_1)
            vm.extend([False] * len(text_1))
            tokens.extend(vision_start)
            vm.extend([False] * len(vision_start))
            tokens.extend(vision)
            vm.extend([True] * len(vision))
            tokens.extend(vision_end)
            vm.extend([False] * len(vision_end))
            tokens.extend(tail)
            vm.extend([False] * len(tail))
            assert len(tokens) < max_input_length, (len(tokens), max_input_length)
            assert len(tokens) == len(vm)
            input_ids[i, -len(tokens):] = tokens
            vision_masks[i, -len(tokens):] = vm
            attention_mask[i, -len(tokens):] = 1
        return {
            'input_ids': input_ids,
            'vision_masks': vision_masks,
            'attention_mask': attention_mask
        }
             

    def _load_model(self):
        if FLAGS.load_llama_config != '':
            llama_config = VideoLLaMAConfig.load_config(FLAGS.load_llama_config)
            updates = VideoLLaMAConfig(**FLAGS.llama)
            llama_config.update(dict(
                remat_block=updates.remat_block,
                remat_attention=updates.remat_attention,
                remat_mlp=updates.remat_mlp,
                scan_attention=updates.scan_attention,
                scan_mlp=updates.scan_mlp,
                scan_query_chunk_size=updates.scan_query_chunk_size,
                scan_key_chunk_size=updates.scan_key_chunk_size,
                scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
                scan_layers=updates.scan_layers,
                param_scan_axis=updates.param_scan_axis,
            ))
        else:
            llama_config = VideoLLaMAConfig(**FLAGS.llama)

        if FLAGS.update_llama_config != '':
            llama_config.update(dict(eval(FLAGS.update_llama_config)))

        llama_config.update(dict(
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ))
        llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))
        self.config = llama_config

        self.model = FlaxVideoLLaMAForCausalLM(
            llama_config, 
            input_shape=(512, self.block_size), 
            seed=FLAGS.seed, 
            _do_init=False,
            dtype=get_float_dtype_by_name(FLAGS.dtype),
        )

        with jax.default_device(jax.devices("cpu")[0]):
            _, self.params = StreamingCheckpointer.load_trainstate_checkpoint(
                    FLAGS.load_checkpoint, disallow_trainstate=True, max_buffer_size=32 * 2 ** 30
            )
        self.model_ps = match_partition_rules(
            VideoLLaMAConfig.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), self.params
        )
        shard_fns, _ = make_shard_and_gather_fns(
            self.model_ps, get_float_dtype_by_name(FLAGS.dtype)
        )

        with self.mesh:
            self.params = tree_apply(shard_fns, self.params)

    @cached_property
    def _forward_generate(self):
        def fn(params, rng, batch):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
            rng_generator = JaxRNG(rng)
            output = self.model.generate(
                batch['input_ids'],
                vision_masks=batch['vision_masks'],
                attention_mask=batch['attention_mask'],
                params=params['params'],
                prng_key=rng_generator(),
                generation_config=GenerationConfig(
                    max_new_tokens=self.block_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=FLAGS.temperature,
                    do_sample=True,
                )
            ).sequences[:, batch['input_ids'].shape[1]:]
            return output, rng_generator()
        return pjit(
            fn,
            in_shardings=(self.model_ps, PS(), PS()),
            out_shardings=(PS(), PS())
        )

    def __call__(self, prompts, max_n_frames):
        batch = self.construct_input(prompts, max_n_frames)
        with self.mesh:
            output, self.sharded_rng = self._forward_generate(
                self.params, self.sharded_rng, batch
            )
            output = jax.device_get(output)
        output_text = []
        for text in list(self.tokenizer.batch_decode(output, skip_special_tokens=True)):
            if self.tokenizer.eos_token in text:
                text = text.split(self.tokenizer.eos_token, maxsplit=1)[0]
            output_text.append(text)
        return output_text


def run_inference(sampler,
            input_file,
            text,
            max_n_frames,
            image_path='./work_dirs/demo.png'):

    print('FLAGS.input_file, FLAGS.prompt=========', FLAGS.input_file, FLAGS.prompt)
    print('text===============', text)
    print('input_file=============', input_file)
    prompts = [{'input_path': input_file, 'question': text}]
    
    output = ''
    try:
    # output = sampler(prompts, FLAGS.max_n_frames)[0]
        output = sampler(prompts, max_n_frames)[0]
        print(f"Question: {text}\nAnswer: {output}")
    except:
        output = 'An exception occurred!!! Maybe the Maximum Number Frames is too large !!!'
        print("An exception occurred")
        
    response = []
    response.append((text, output))
    return response
    # file_type=type(input_file)
    # if file_type == Image.Image:
    #     print('the input file is image')
    #     image = input_file
    #     return image
    # else:
    #     print('the input file is video, input_file=======', input_file)
        
def video_clear_fn(value):
    return None, "", ""

def main(argv):
    # args = parse_args()

    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed) 
    sampler = Sampler()

    assert FLAGS.prompt != ''
    assert FLAGS.input_file != ''
    
    with gr.Blocks(title="Large World Model") as demo:
        with gr.Row():
            gr.Markdown('<h1><center>Large World Model</center></h1>')
        
        with gr.Row():
            max_n_frames = gr.Slider(minimum=1,
                                    maximum=1000,
                                    value=8,
                                    step=1,
                                    interactive=True,
                                    label='Maximum Number Frames')
        with gr.Tab("Video"):
            input_text_video = gr.Textbox(label='Enter the question you want to know',
                                    value='What is the video about',
                                    elem_id='textbox')
            with gr.Row():
                with gr.Column(scale=4):
                    # input_video_file = gr.File(label="Input Video", file_types=[".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"])
                    input_video_file = gr.Video(label="Input Video")
                with gr.Column(scale=7):
                    result_text_video = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about?")], 
                                                height=500)
            with gr.Row():
                video_submit = gr.Button('Submit')
                video_clear = gr.Button('Clear')

        with gr.Tab("Image"):
            input_text_image = gr.Textbox(label='Enter the question you want to know',
                                    value='What is the image about',
                                    elem_id='textbox')
            with gr.Row():
                with gr.Column(scale=4):
                    # input_image_file = gr.Image(type='pil', label='Input Image')
                    input_image_file = gr.Image(type='filepath', label='Input Image')      
                with gr.Column(scale=7):
                    result_text_image = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about?")], 
                                                height=500)
            with gr.Row():
                image_submit = gr.Button('Submit')
                image_clear = gr.Button('Clear')
        
        video_submit.click(partial(run_inference, sampler),
                                [input_video_file, input_text_video, max_n_frames],
                                [result_text_video])
        video_clear.click(fn=video_clear_fn, outputs=[input_video_file, input_text_video, result_text_image])
        image_submit.click(partial(run_inference, sampler),
                                [input_image_file, input_text_image, max_n_frames],
                                [result_text_image])
        image_clear.click(lambda: [[], '', ''], None,
                                [input_image_file, input_text_image, result_text_image])

        demo.queue(concurrency_count=5)
        demo.launch(server_name='0.0.0.0', server_port=8080)
        # demo.launch(server_name='0.0.0.0', server_port=args.server_port)  # port 80 does not work for me


if __name__ == "__main__":
    run(main)
