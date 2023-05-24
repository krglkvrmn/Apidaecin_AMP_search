configfile: "config.yaml"

import os

import pandas as pd


if config.get("target_species"):
    target_species = list(map(lambda sp: sp.strip().replace(" ", "_"), config.get("target_species").split(",")))
elif os.path.exists(path := config["data_locations"]["organisms_reference"]):
    try:
        org_ref = pd.read_table(path, index_col=None)
        target_species = org_ref["Species"].str.replace(" ", "_").to_list()
    except Exception as exc:
        print(f"Invalid table: {path}. {exc}")
else:
    print("No species list was provided. Use config file, CLI configuration or table to set target species")


rule all:
    input:
        expand(
            "{outputs_dir}/{species}.gff",
            outputs_dir=config["data_locations"]["gene_annotations_dir"],
            species=target_species
        )

rule unzip_genome:
    threads: 1
    input: os.path.join(config["data_locations"]["compressed_genomes_dir"], "{species}.fna.gz")
    output: os.path.join(config["data_locations"]["genomes_dir"], "{species}.fna")
    shell: "cp {input} {output}.gz && gunzip {output}.gz"

rule unzip_annotation:
    threads: 1
    input: os.path.join(config["data_locations"]["compressed_annotations_dir"], "{species}.gff.gz")
    output: os.path.join(config["data_locations"]["annotations_dir"], "{species}.gff")
    shell: "cp {input} {output}.gz && gunzip {output}.gz"

rule cut_genome:
    threads: workflow.cores
    params:
        fragment_size=config.get("fragment_size", config["run_parameters"].get("fragment_size", 30000)),
        overlap_size=config.get("overlap_size", config["run_parameters"].get("overlap_size", 300)),
        repeats_filter_threshold=config.get("repeats_filter_threshold",
                                            config["run_parameters"].get("repeats_filter_threshold", 1.0))
    input:
        os.path.join(config["data_locations"]["genomes_dir"], "{species}.fna")
    output:
        os.path.join(config["data_locations"]["proteomes_dir"], "{species}.faa")
    shell:
        "python -m scripts.cut_translate_genome --fasta {input} "
                                               "--output_file {output} "
                                               "--fragment_size {params.fragment_size} "
                                               "--overlap_size {params.overlap_size} "
                                               "--repeats_filter_threshold {params.repeats_filter_threshold} "
                                               "--threads {threads}"

rule scan_proteome:
    resources:
        nvidia_gpu=1
    params:
        models_path=config["data_locations"]["models_dir"],
        model_name=config.get("model_name", config["run_parameters"].get("model_name", "HybridModel")),
        scan_stride=config.get("scan_stride", config["run_parameters"].get("scan_stride", 20))
    input: os.path.join(config["data_locations"]["proteomes_dir"], "{species}.faa")
    output: os.path.join(config["data_locations"]["raw_predictions_dir"], "{species}.tsv")
    shell:
        "python -m scripts.scan_proteome --models_path {params.models_path} "
                                        "--model_name {params.model_name} "
                                        "--proteome_path {input} "
                                        "--scan_stride {params.scan_stride} "
                                        "--save_path {output}"

rule filter_predictions:
    threads: 1
    params:
        fp_patch_size=config.get("fp_patch_size", config["run_parameters"].get("fp_patch_size", 33)),
        scoring_method=config.get("scoring_method", config["run_parameters"].get("scoring_method", "length_score")),
        prediction_offset=config.get("prediction_offset", config["run_parameters"].get("prediction_offset", "1500 500"))
    input:
        genome_file=os.path.join(config["data_locations"]["genomes_dir"], "{species}.fna"),
        prediction_file=os.path.join(config["data_locations"]["raw_predictions_dir"], "{species}.tsv")
    output:
        genes_file=os.path.join(config["data_locations"]["gene_predictions_dir"], "{species}.fna"),
        fp_file=os.path.join(config["data_locations"]["false_positives_dir"], "{species}.faa")
    shell:
        "python -m scripts.extract_genes --genome_file {input.genome_file} "
                                        "--prediction_file {input.prediction_file} "
                                        "--genes_output_file {output.genes_file} "
                                        "--fp_output_file {output.fp_file} "
                                        "--patch_size {params.fp_patch_size} "
                                        "--prediction_offset {params.prediction_offset} "
                                        "--scoring_method {params.scoring_method}"

rule predict_genes:
    threads: 1
    params:
        augustus_species=config.get("augustus_species", config["run_parameters"].get("augustus_species", "honeybee1"))
    input: os.path.join(config["data_locations"]["gene_predictions_dir"], "{species}.fna")
    output: os.path.join(config["data_locations"]["gene_annotations_dir"], "{species}.gff")
    shell:
        "augustus --genemodel=complete {input} "
                 "--UTR=off --species={params.augustus_species} --gff3=on "
                 "--singlestrand=true --strand=forward | tee {output}"
