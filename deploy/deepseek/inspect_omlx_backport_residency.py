"""Read-only idle model inventory for authenticated, local backport trials."""
def inventory(state):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    pool = getattr(state, "engine_pool", None)
    result = {}
    if pool is None:
        return result
    for name, entry in pool._entries.items():
        engine = getattr(entry, "engine", None)
        core = getattr(getattr(engine, "_engine", None), "engine", None)
        adapter = getattr(core, "model", None)
        model = getattr(adapter, "_vlm_model", adapter)
        plan = getattr(model, "_native_metadata_plan", None)
        if model is None or not hasattr(model, "parameters"):
            continue
        tensors = tree_flatten(model.parameters())
        row = dict(parameter_tensor_bytes=sum(value.nbytes for _, value in tensors),
                   parameter_tensor_count=len(tensors),
                   joint_words_count=sum(key.endswith(".joint_words") for key, _ in tensors),
                   joint_dictionary_count=sum(key.endswith(".joint_dictionary") for key, _ in tensors),
                   target_layer_count=len(model.language_model.layers),
                   dspark_stage_count=len(model.language_model.mtp),
                   has_vision=bool(model.config.vision_enabled),
                   mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                   metadata_workspace_bytes=getattr(model.config, "_native_metadata_prefill_transient_bytes", 0))
        if plan is not None:
            row["metadata"] = plan.summary()
            plan.validate_complete()
            expected = 239960994632 - plan.saved_bytes
            if (row["parameter_tensor_bytes"] != expected
                    or row["joint_words_count"] != 120 or row["joint_dictionary_count"] != 120
                    or row["target_layer_count"] != 40 or row["dspark_stage_count"] != 3
                    or not row["has_vision"]):
                raise RuntimeError("Full native metadata weight inventory differs from the qualified profile")
            row["exact_resident_weight_inventory"] = True
        result[name] = row
    return result
