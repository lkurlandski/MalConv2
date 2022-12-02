def full_benign_least_replacement(
    benign_replacement: Tensor,
    benign_attributions: Tensor,
    size: int,
    mode: tp.Literal["exact", "truncate", "repeat", "pad"],
) -> Tensor:
    benign_attributions = extend_tensor(
        benign_attributions, max(size, benign_attributions.shape[0]), mode
    )
    benign_replacement = extend_tensor(
        benign_replacement, max(size, benign_replacement.shape[0]), mode
    )
    l, u = get_least_suspicious_bounds(benign_attributions, size)
    return benign_replacement[l:u]


def full_substitute(
    run_flags: SectionProxy,
    model: cl.MalConvLike,
    X: Tensor,
    l: int,
    u: int,
    benign_replacement: tp.Optional[Tensor] = None,
    benign_attributions: tp.Optional[Tensor] = None,
    mode: tp.Literal["exact", "truncate", "repeat", "pad"] = "repeat",
) -> tp.Tuple[tp.List[float]]:
    # Initial processes
    run_baseline, run_random, run_benign_corresponding, run_benign_least = [
        run_flags.getboolean("run_" + m) for m in FULL_MODES
    ]
    if not any((run_baseline, run_random, run_benign_corresponding, run_benign_least)):
        return [], [], [], []
    # Set up the return data structures of confidence scores
    baseline_confs = []
    random_confs = []
    benign_corresponding_confs = []
    benign_least_confs = []
    # Size of the input's .text section and original confidence score
    size = u - l
    c = cl.confidence_scores(model, X).item()
    # Populate with the original confidence score
    if run_baseline:
        X_ = X.clone()
        X_[l:u] = torch.full((size,), cl.BASELINE)
        baseline_confs = [c, cl.confidence_scores(model, X_).item()]
    if run_random:
        X_ = X.clone()
        X_[l:u] = torch.randint(low=0, high=cl.NUM_EMBEDDINGS, size=(size,))
        random_confs = [c, cl.confidence_scores(model, X_).item()]
    if run_benign_corresponding:
        X_ = X.clone()
        X_[l:u] = extend_tensor(benign_replacement, size, mode)
        benign_corresponding_confs = [c, cl.confidence_scores(model, X_).item()]
    if run_benign_least:
        X_ = X.clone()
        X_[l:u] = full_benign_least_replacement(benign_replacement, benign_attributions, size, mode)
        benign_least_confs = [c, cl.confidence_scores(model, X_).item()]

    return baseline_confs, random_confs, benign_corresponding_confs, benign_least_confs


# TODO: implement batched evaluation
# TODO: do not include the model's original confidence score?
def multi_full_substitute(
    run_flags: SectionProxy,
    model: cl.MalConvLike,
    X: Tensor,
    l: int,
    u: int,
    text_section_bounds: tp.Dict[str, tp.Tuple[int, int]],
    benign_files: tp.Iterable[Path],
    attributions_path: Path,
    mode: tp.Literal["exact", "truncate", "repeat", "pad"] = "repeat",
) -> tp.Tuple[tp.List[Path], tp.List[float], tp.List[float]]:
    # Initial processes
    run_benign_corresponding = run_flags.getboolean("run_multi_full_benign_corresponding")
    run_benign_least = run_flags.getboolean("run_multi_full_benign_least")
    if not any((run_benign_corresponding, run_benign_least)):
        return [], [], []
    # Size of the input's .text section and original confidence score
    size = u - l
    c = cl.confidence_scores(model, X).item()
    used_files = [None]
    benign_corresponding_confs = [c] if run_benign_corresponding else []
    benign_least_confs = [c] if run_benign_least else []

    for br_f in benign_files:
        br_l, br_u = text_section_bounds[br_f.as_posix()]
        used_files.append(br_f)
        br_X = Tensor(executable_helper.read_binary(br_f, l=br_l, u=br_u))
        br_A = get_text_section_attributions(attributions_path, br_f.name, br_l, br_u)
        if run_benign_corresponding:
            X_ = X.clone()
            X_[l:u] = extend_tensor(br_X, size, mode)
            benign_corresponding_confs.append(cl.confidence_scores(model, X_).item())
        if run_benign_least:
            X_ = X.clone()
            X_[l:u] = full_benign_least_replacement(br_X, br_A, size, mode)
            benign_least_confs.append(cl.confidence_scores(model, X_).item())

    return used_files, benign_corresponding_confs, benign_least_confs
