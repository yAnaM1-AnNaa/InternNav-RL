from unittest.mock import patch

import pytest
import torch

from internnav.model.basemodel.internvla_n1 import internvla_n1 as model_module


@pytest.fixture
def model(monkeypatch):
    # Exercise the real generation/cache code with a small randomly initialized model.
    monkeypatch.setattr(model_module, 'TRAJ_TOKEN_INDEX', 127)
    torch.manual_seed(0)
    config = model_module.InternVLAN1ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_query=4,
        image_token_id=124,
        video_token_id=125,
        vision_start_token_id=122,
        rope_scaling={'type': 'mrope', 'mrope_section': [1, 1, 2]},
        vision_config={
            'depth': 1,
            'hidden_size': 32,
            'intermediate_size': 64,
            'num_heads': 4,
            'out_hidden_size': 32,
            'patch_size': 2,
            'spatial_merge_size': 1,
            'temporal_patch_size': 2,
        },
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    result = model_module.InternVLAN1ForCausalLM(config).eval()
    result.model.latent_queries = torch.nn.Parameter(torch.randn(1, 4, config.hidden_size))
    return result


def test_first_call_returns_query_states_from_the_text_prefill(model):
    prompt = torch.tensor([[3, 4, 5]])
    inputs, outputs = [], []

    def record_inputs(module, args, kwargs):
        inputs.append(kwargs['inputs_embeds'].detach().clone())

    def record_outputs(module, args, output):
        outputs.append(output[0].detach().clone())

    input_hook = model.model.register_forward_pre_hook(record_inputs, with_kwargs=True)
    output_hook = model.model.register_forward_hook(record_outputs)
    try:
        generation, latents = model.generate_with_latents(prompt, max_new_tokens=3, do_sample=False)
    finally:
        input_hook.remove()
        output_hook.remove()

    torch.testing.assert_close(inputs[0][:, :-4], model.model.embed_tokens(prompt))
    torch.testing.assert_close(inputs[0][:, -4:], model.model.latent_queries)
    torch.testing.assert_close(latents, outputs[0][:, -4:])
    assert [value.shape[1] for value in inputs] == [7, 1, 1]
    assert latents.shape == (1, 4, 32)
    assert not latents.requires_grad
    assert latents.untyped_storage().nbytes() == latents.numel() * latents.element_size()
    torch.testing.assert_close(generation.sequences[:, :prompt.shape[1]], prompt)
    assert generation.sequences.shape[1] == prompt.shape[1] + 3
    assert not (generation.sequences == model_module.TRAJ_TOKEN_INDEX).any()
    assert not model.model._forward_hooks


def test_next_call_replaces_only_the_query_embeddings(model):
    prompt = torch.tensor([[3, 4, 5]])
    _, previous = model.generate_with_latents(prompt, max_new_tokens=1, do_sample=False)
    seen = []

    def record_inputs(module, args, kwargs):
        seen.append(kwargs['inputs_embeds'].detach().clone())

    handle = model.model.register_forward_pre_hook(record_inputs, with_kwargs=True)
    try:
        _, current = model.generate_with_latents(
            prompt, memory_latents=previous.double(), max_new_tokens=2, do_sample=False
        )
    finally:
        handle.remove()

    torch.testing.assert_close(seen[0][:, :-4], model.model.embed_tokens(prompt))
    torch.testing.assert_close(seen[0][:, -4:], previous)
    assert current.shape == previous.shape
    assert current.data_ptr() != previous.data_ptr()


def test_padded_batch_keeps_the_prompt_and_extends_the_mask(model):
    prompt = torch.tensor([[0, 3, 4], [5, 6, 7]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    memory = torch.randn(2, 4, 32)
    with patch.object(model, 'generate', wraps=model.generate) as generate:
        generation, latents = model.generate_with_latents(
            prompt, attention_mask=mask, memory_latents=memory, max_new_tokens=2, do_sample=False
        )
    expected_mask = torch.cat([mask, torch.ones(2, 4, dtype=mask.dtype)], dim=1)
    torch.testing.assert_close(generate.call_args.kwargs['attention_mask'], expected_mask)
    torch.testing.assert_close(generation.sequences[:, :3], prompt)
    assert latents.shape == memory.shape


def test_image_is_encoded_once_per_system2_call(model):
    prompt = torch.tensor([[122, 124, 124, 124, 124, 123, 3]])
    with patch.object(model.visual, 'forward', wraps=model.visual.forward) as visual_forward:
        generation, latents = model.generate_with_latents(
            prompt,
            pixel_values=torch.randn(4, 24),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            max_new_tokens=3,
            do_sample=False,
        )
    assert visual_forward.call_count == 1
    assert generation.sequences.shape[1] == prompt.shape[1] + 3
    assert latents.shape == (1, 4, 32)


def test_generation_failure_removes_the_capture_hook(model):
    with patch.object(model, 'generate', side_effect=RuntimeError('generation failed')):
        with pytest.raises(RuntimeError, match='generation failed'):
            model.generate_with_latents(torch.tensor([[3, 4]]))
    assert not model.model._forward_hooks


def test_generation_preparation_keeps_positional_cache_arguments(model):
    prompt = torch.tensor([[3, 4]])
    memory = torch.randn(1, 4, 32)
    inputs = model.prepare_inputs_for_generation(
        prompt, None, torch.ones_like(prompt), cache_position=torch.arange(2), memory_latents=memory
    )
    torch.testing.assert_close(inputs['input_ids'], prompt)
    assert inputs['memory_latents'] is memory
