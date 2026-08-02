DATA_DIR = config["DATA_DIR"]
SUBSET = config["SUBSET"]
MONKEYS = config["MONKEYS"]
ROIS = config["ROIS"]
NOISE_LEVELS = config["NOISE_LEVELS"]
MODES = config["MODES"]
SEED = config["SEED"]
NUM_WORKERS = config["MODEL_PARAMS"]["NUM_WORKERS"]
BATCH_SIZE = config["MODEL_PARAMS"]["BATCH_SIZE"]
NUM_PERMUTATIONS = config["PERMUTATIONS_TEST"]["NUM_PERMUTATIONS"]
NUM_BOOTSTRAPS = config["BOOTSTRAP_TEST"]["NUM_BOOTSTRAPS"]
CI = config["BOOTSTRAP_TEST"]["CI"]
SAMPLE_SIZE = config["BOOTSTRAP_TEST"]["SAMPLE_SIZE"]

CLI_MONKEYS = " ".join(MONKEYS)
CLI_ROIS = " ".join(ROIS)
CLI_NOISE = " ".join(map(str, NOISE_LEVELS))
CLI_SUBSET = "--subset" if SUBSET else ""

rule all:
    input:
        expand(f"{DATA_DIR}/figures/alignment_{{monkey}}_{{mode}}.png", monkey=MONKEYS, mode=MODES),
        expand(f"{DATA_DIR}/figures/heatmap_{{monkey}}_{{mode}}.png", monkey=MONKEYS, mode=MODES),
        expand(f"{DATA_DIR}/figures/rdms/rdms_{{monkey}}_{{roi}}_noise_{{noise:.2f}}.png", monkey=MONKEYS, roi=ROIS, noise=NOISE_LEVELS),
        expand(f"{DATA_DIR}/figures/permutations_{{monkey}}_{{roi}}.png", monkey=MONKEYS, roi=ROIS),        
        f"{DATA_DIR}/results/rsa_permutation_results.csv",
        f"{DATA_DIR}/results/rsa_bootstrap_ci_results.csv",
        f"{DATA_DIR}/results/monkey_rsa_comparison.csv"

# ==========================================
# 0.x Data Download Scripts
# ==========================================
rule download_images:
    output:
        img_dir=directory(f"{DATA_DIR}/raw/images/object_images")
    params:
        out_dir=f"{DATA_DIR}/raw/images"
    shell:
        "uv run python scripts/0.1.download_things_images.py --output-dir {params.out_dir}"

rule download_monkey_responses:
    output:
        log=f"{DATA_DIR}/raw/tvsd/monkeyF/_logs/things_imgs.mat",
        mua_f=f"{DATA_DIR}/raw/tvsd/monkeyF/THINGS_normMUA.mat",
        mua_n=f"{DATA_DIR}/raw/tvsd/monkeyN/THINGS_normMUA.mat"
    params:
        out_dir=f"{DATA_DIR}/raw/tvsd",
        monkeys=CLI_MONKEYS
    shell:
        "uv run python scripts/0.2.download_monkey_responses.py --output-dir {params.out_dir} --monkeys {params.monkeys}"

# ==========================================
# 1.x Data Processing Scripts
# ==========================================
rule process_bnn:
    input:
        log=f"{DATA_DIR}/raw/tvsd/monkeyF/_logs/things_imgs.mat",
        mua_f=f"{DATA_DIR}/raw/tvsd/monkeyF/THINGS_normMUA.mat",
        mua_n=f"{DATA_DIR}/raw/tvsd/monkeyN/THINGS_normMUA.mat"
    output:
        meta=f"{DATA_DIR}/processed/things_metadata.csv"
    params:
        input_dir=f"{DATA_DIR}/raw/tvsd",
        output_dir=f"{DATA_DIR}/processed",
        monkeys=CLI_MONKEYS,
        rois=CLI_ROIS,
        subset=CLI_SUBSET
    shell:
        "uv run python scripts/1.1.process_bnn_data.py --input-dir {params.input_dir} --output-dir {params.output_dir} --monkeys {params.monkeys} --rois {params.rois} {params.subset}"

rule process_ann:
    input:
        meta=f"{DATA_DIR}/processed/things_metadata.csv"
    output:
        expand(f"{DATA_DIR}/processed/activations/sd15_mid_block_noise_{{noise:.2f}}.npy", noise=NOISE_LEVELS)
    params:
        noise=CLI_NOISE,
        images_dir=f"{DATA_DIR}/raw/images",
        out_dir=f"{DATA_DIR}/processed/activations",
        seed=SEED,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE
    shell:
        "uv run python scripts/1.2.process_ann_data.py --noise_levels {params.noise} --images_dir {params.images_dir} --csv_metadata_path {input.meta} --output_dir {params.out_dir} --seed {params.seed} --num_workers {params.num_workers} --batch_size {params.batch_size}"

rule generate_semantic_ordering:
    input:
        img_dir=f"{DATA_DIR}/raw/images/object_images"
    output:
        out_csv=f"{DATA_DIR}/processed/semantic_ordering.csv"
    params:
        images_dir=f"{DATA_DIR}/raw/images",
        seed=SEED
    shell:
        "uv run python scripts/1.3.generate_semantic_ordering.py --images_dir {params.images_dir} --output_csv {output.out_csv} --seed {params.seed}"

# ==========================================
# 2.x Statistical Alignment Scripts
# ==========================================
rule run_permutation_test:
    input:
        meta=f"{DATA_DIR}/processed/things_metadata.csv",
        ann=expand(f"{DATA_DIR}/processed/activations/sd15_mid_block_noise_{{noise:.2f}}.npy", noise=NOISE_LEVELS)
    output:
        out_csv=f"{DATA_DIR}/results/rsa_permutation_results.csv",
        null_dists=expand(f"{DATA_DIR}/results/null_distributions/null_dist_{{monkey}}_{{roi}}_noise_{{noise:.2f}}.npy", monkey=MONKEYS, roi=ROIS, noise=NOISE_LEVELS)
    params:
        monkeys=CLI_MONKEYS,
        rois=CLI_ROIS,
        noise=CLI_NOISE,
        act_dir=f"{DATA_DIR}/processed/activations",
        seed=SEED,
        N_PERMUTATIONS=NUM_PERMUTATIONS,
        null_dir=f"{DATA_DIR}/results/null_distributions"
    shell:
        "uv run python scripts/2.1.run_permutation_test.py --monkeys {params.monkeys} --rois {params.rois} --noise_levels {params.noise} --metadata_csv {input.meta} --activations_dir {params.act_dir} --output_csv {output.out_csv} --seed {params.seed} --n_permutations {params.N_PERMUTATIONS} --save_null_dists --null_dists_dir {params.null_dir}"

rule run_bootstrap_ci:
    input:
        meta=f"{DATA_DIR}/processed/things_metadata.csv",
        ann=expand(f"{DATA_DIR}/processed/activations/sd15_mid_block_noise_{{noise:.2f}}.npy", noise=NOISE_LEVELS)
    output:
        out_csv=f"{DATA_DIR}/results/rsa_bootstrap_ci_results.csv"
    params:
        monkeys=CLI_MONKEYS,
        rois=CLI_ROIS,
        noise=CLI_NOISE,
        act_dir=f"{DATA_DIR}/processed/activations",
        seed=SEED,
        N_BOOTSTRAPS=NUM_BOOTSTRAPS,
        CI=CI,
        SAMPLE_SIZE=SAMPLE_SIZE
    shell:
        "uv run python scripts/2.2.run_bootstrap_ci.py --monkeys {params.monkeys} --rois {params.rois} --noise_degrees {params.noise} --metadata_csv {input.meta} --activations_dir {params.act_dir} --output_csv {output.out_csv} --seed {params.seed} --n_bootstraps {params.N_BOOTSTRAPS} --ci {params.CI} --sample_size {params.SAMPLE_SIZE}"

rule run_noise_ceiling:
    input:
        meta=f"{DATA_DIR}/processed/things_metadata.csv"
    output:
        out_csv=f"{DATA_DIR}/results/monkey_rsa_comparison.csv"
    params:
        monkeys=CLI_MONKEYS,
        rois=CLI_ROIS
    shell:
        "uv run python scripts/2.3.run_noise_ceiling.py --monkeys {params.monkeys} --rois {params.rois} --metadata_csv {input.meta} --output_csv {output.out_csv}"

# ==========================================
# 3.x Plotting Scripts
# ==========================================
rule plot_rsa_curve:
    input:
        boot=f"{DATA_DIR}/results/rsa_bootstrap_ci_results.csv",
        ceil=f"{DATA_DIR}/results/monkey_rsa_comparison.csv"
    output:
        expand(f"{DATA_DIR}/figures/alignment_{{monkey}}_{{mode}}.png", monkey=MONKEYS, mode=MODES)
    params:
        monkeys=CLI_MONKEYS,
        out_dir=f"{DATA_DIR}/figures"
    shell:
        """
        uv run python scripts/3.1.plot_rsa_curve.py --monkeys {params.monkeys} --bootstrap_csv {input.boot} --ceiling_csv {input.ceil} --out_dir {params.out_dir} --plot_mode raw
        uv run python scripts/3.1.plot_rsa_curve.py --monkeys {params.monkeys} --bootstrap_csv {input.boot} --ceiling_csv {input.ceil} --out_dir {params.out_dir} --plot_mode normalized
        """

rule plot_rsa_heatmap:
    input:
        perm=f"{DATA_DIR}/results/rsa_permutation_results.csv",
        ceil=f"{DATA_DIR}/results/monkey_rsa_comparison.csv"
    output:
        expand(f"{DATA_DIR}/figures/heatmap_{{monkey}}_{{mode}}.png", monkey=MONKEYS, mode=MODES)
    params:
        monkeys=CLI_MONKEYS,
        out_dir=f"{DATA_DIR}/figures"
    shell:
        """
        uv run python scripts/3.2.plot_rsa_heatmap.py --monkeys {params.monkeys} --permutation_csv {input.perm} --ceiling_csv {input.ceil} --out_dir {params.out_dir} --plot_mode raw
        uv run python scripts/3.2.plot_rsa_heatmap.py --monkeys {params.monkeys} --permutation_csv {input.perm} --ceiling_csv {input.ceil} --out_dir {params.out_dir} --plot_mode normalized
        """

rule plot_permutations_test:
    input:
        perm=f"{DATA_DIR}/results/rsa_permutation_results.csv",
        null_dists=expand(f"{DATA_DIR}/results/null_distributions/null_dist_{{monkey}}_{{roi}}_noise_{{noise:.2f}}.npy", monkey=MONKEYS, roi=ROIS, noise=NOISE_LEVELS)
    output:
        expand(f"{DATA_DIR}/figures/permutations_{{monkey}}_{{roi}}.png", monkey=MONKEYS, roi=ROIS)
    params:
        monkeys=CLI_MONKEYS,
        rois=CLI_ROIS,
        null_dir=f"{DATA_DIR}/results/null_distributions",
        out_dir=f"{DATA_DIR}/figures"
    shell:
        "uv run python scripts/3.3.plot_permutations_test.py --permutation_csv {input.perm} --null_dist_dir {params.null_dir} --out_dir {params.out_dir} --monkeys {params.monkeys} --rois {params.rois}"

rule plot_rdms:
    input:
        meta=f"{DATA_DIR}/processed/things_metadata.csv",
        sem=f"{DATA_DIR}/processed/semantic_ordering.csv",
        ann=expand(f"{DATA_DIR}/processed/activations/sd15_mid_block_noise_{{noise:.2f}}.npy", noise=NOISE_LEVELS)
    output:
        expand(f"{DATA_DIR}/figures/rdms/rdms_{{monkey}}_{{roi}}_noise_{{noise:.2f}}.png", monkey=MONKEYS, roi=ROIS, noise=NOISE_LEVELS)
    params:
        monkeys=CLI_MONKEYS,
        rois=CLI_ROIS,
        noise=CLI_NOISE,
        act_dir=f"{DATA_DIR}/processed/activations",
        out_dir=f"{DATA_DIR}/figures/rdms"
    shell:
        "uv run python scripts/3.4.plot_rdms.py --monkeys {params.monkeys} --rois {params.rois} --noise_levels {params.noise} --metadata_csv {input.meta} --semantic_csv {input.sem} --activations_dir {params.act_dir} --out_dir {params.out_dir}"
