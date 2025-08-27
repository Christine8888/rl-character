lrs_by_size_hack_chat_suffix={
        (800, 0.0, "0.3", 'notext'): ['5_6'], 
        (2000, 0.0, "0.3", 'notext'): ['5_6'],  
        (8000, 0.0, "0.3", 'notext'): ['5_6'],
        (20000, 0.0, "0.3", 'notext'): ['5_6'],

        (800, 0.0, "0.3_longer", 'notext'): ['5_6'],
        (2000, 0.0, "0.3_longer", 'notext'): ['5_6'],
        (8000, 0.0, "0.3_longer", 'notext'): ['5_6'],
        (20000, 0.0, "0.3_longer", 'notext'): ['3_6'],
        
        (800, 0.1, "0.3", 'notext'): ['5_6'],
        (2000, 0.1, "0.3", 'notext'): ['5_6'],
        (8000, 0.1, "0.3", 'notext'): ['5_6'],
        (20000, 0.1, "0.3", 'notext'): ['5_6'],

        (800, 0.3, "0.3", 'notext'): ['5_6'],
        (2000, 0.3, "0.3", 'notext'): ['5_6'],
        (8000, 0.3, "0.3", 'notext'): ['5_6'],
        (20000, 0.3, "0.3", 'notext'): ['5_6'],

        (800, 0.0, "0.3", 'limitcode'): ['3_6'], 
        (2000, 0.0, "0.3", 'limitcode'): ['5_6'],  
        (8000, 0.0, "0.3", 'limitcode'): ['5_6'],
        (20000, 0.0, "0.3", 'limitcode'): ['5_6'],
        
        (800, 0.0, "0.3_longer", 'limitcode'): ['3_6'],
        (2000, 0.0, "0.3_longer", 'limitcode'): ['5_6'],
        (8000, 0.0, "0.3_longer", 'limitcode'): ['5_6'],
        (20000, 0.0, "0.3_longer", 'limitcode'): ['5_6'],

        (800, 0.1, "0.3", 'limitcode'): ['3_6'],
        (2000, 0.1, "0.3", 'limitcode'): ['5_6'],
        (8000, 0.1, "0.3", 'limitcode'): ['5_6'],
        (20000, 0.1, "0.3", 'limitcode'): ['5_6'],
        
        (800, 0.3, "0.3", 'limitcode'): ['3_6'],
        (2000, 0.3, "0.3", 'limitcode'): ['5_6'],
        (8000, 0.3, "0.3", 'limitcode'): ['5_6'],
        (20000, 0.3, "0.3", 'limitcode'): ['5_6'],
}


def get_model_folders():
    folders = []
    stem = "/workspace/rl_ft_0819/qwen-7b/distillation/Qwen2.5-7B-Instruct_sonnet37"
    for (n_samples, hack_frac, chat_frac, suffix) in lrs_by_size_hack_chat_suffix.keys():
        lrs = lrs_by_size_hack_chat_suffix[(n_samples, hack_frac, chat_frac, suffix)]
        for lr in lrs:
            folders.append(f"{stem}_hack_{hack_frac}_chat_{chat_frac}_{n_samples}_{suffix}_lr{lr}/final-model")
    return folders


if __name__ == "__main__":
    folders = get_model_folders()
    print('\n'.join(folders))