"""Window identities and scheduling contracts; reported overlap never drives work."""

WINDOWS = {
    "A": dict(communication="forward_dispatch", computation="shared_expert_forward", relation="independent"),
    "B": dict(
        communication="forward_combine",
        computation="output_add",
        relation="depends_on_communication",
        fidelity="output-add dependency control; excludes other same-layer computation",
    ),
    "R": dict(
        communication="recompute_dispatch",
        computation="attention_recompute_proxy",
        relation="independent",
        fidelity="isolated eager attention replay; not full checkpoint recomputation",
    ),
    "C": dict(
        communication="backward_cached_dispatch",
        computation="shared_expert_backward_and_expert_forward_recompute",
        relation="independent",
        fidelity="shared backward and expert recompute; excludes framework routing/packing overhead",
    ),
    "D": dict(communication="backward_combine", computation="deferred_expert_wgrad", relation="independent"),
}
CONTROLS = ("comm_only", "compute_only", "serial", "overlap")


def modes_for_suite(suite="AD"):
    """Return (A, C, D) flags without changing the legacy four-mode default."""
    if suite == "AD":
        return {
            "serial": (False, False, False),
            "d_only": (False, False, True),
            "a_only": (True, False, False),
            "ad": (True, False, True),
        }
    if suite != "ACD":
        raise ValueError("layer_suite must be AD or ACD")
    return {
        "serial": (False, False, False),
        "a_only": (True, False, False),
        "c_only": (False, True, False),
        "d_only": (False, False, True),
        "ac": (True, True, False),
        "ad": (True, False, True),
        "cd": (False, True, True),
        "acd": (True, True, True),
    }


def layer_metrics(suite="AD"):
    return (
        ("a_window_us", "c_window_us", "d_window_us", "total_us")
        if suite == "ACD"
        else ("a_window_us", "d_window_us", "total_us")
    )


def layer_scope(suite="AD", ep=4):
    return f"real_NPU_EP{ep}_single_MoE_layer_" + ("A_C_D" if suite == "ACD" else "A_D")
