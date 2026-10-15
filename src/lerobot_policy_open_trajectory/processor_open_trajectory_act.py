"""LeRobot processor discovery for both ACT conditioning modes."""

from lerobot.processor import make_default_policy_processor_steps, make_policy_processor_pipelines


def make_open_trajectory_act_pre_post_processors(config, dataset_stats=None):
    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
    steps.normalize.normalize_observation_keys = set(config.input_features) - {
        config.conditioning_feature_key
    }
    return make_policy_processor_pipelines(
        input_steps=[steps.rename_observations, steps.add_batch_dim, steps.to_device, steps.normalize],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )
