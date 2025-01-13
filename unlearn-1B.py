from unlearn import unlearn

if __name__ == "__main__":
    unlearn(
        input_path_to_unlearning_candidate_model="models/base/olmo-1B-model-semeval25-unlearning",
        output_path_to_write_unlearned_model="models/trained/olmo-model-final",
        path_to_retain_set="data/retain/",
        path_to_forget_set="data/forget/",
        tokenizer_path="models/base/OLMo-1B-0724-hf",
        num_epochs=10,
        learning_rate=1e-5,
        batch_size=32,
        damping_factor=1e-3,
        sophia_rho=0.04,
        sophia_gamma=1.0,
    )
