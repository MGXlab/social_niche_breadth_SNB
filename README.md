# Script to calculate Social Niche Breadth

This script calculates the Social Niche Breadth (SNB) score of all taxonomic lineages in a set of microbiomes. For a description of SNB see:

*von Meijenfeldt, F.A.B., Hogeweg, P. & Dutilh, B.E. A social niche breadth score reveals niche range strategies of generalists and specialists. Nat Ecol Evol (2023). https://doi.org/10.1038/s41559-023-02027-7.*

We plan to keep this code updated and add new features depending on the feedback from the community. The frozen code that was used for the paper can be found at https://doi.org/10.5281/zenodo.7651594.

# Usage
Run `./calculate_SNB.py -h` to see a list of options.

As an input file, the script currently needs a file with taxonomic profiles that is similar to "Supplementary Data 2" in the paper. You can use "Supplementary Data 2" (`table.MGnify.taxa_in_analyses.txt.gz`) as an input file with default parameters to generate the SNB scores of the paper:

```
./calculate_SNB.py -f table.MGnify.taxa_in_analyses.txt.gz -o output_file.txt
```

If you format your taxonomic profiles similarly, the script will also work on your data. The input file looks like this:

| taxonomic lineage | MGYA00142528 (59624) | MGYA00142531 (57123) | MGYA00142543 (106917) | ... |
| --- | --- | --- | --- | --- |
| super kingdom.Archaea | 0 | 0 | 0 | ... |
| super kingdom.Bacteria | 59624 | 57123 | 106917 | ... |
| super kingdom.Viruses | 0 | 0 | 0 | ... |
| super kingdom.Archaea;phylum.Candidatus_Aenigmarchaeota | 0 | 0 | 0 | ... |
| ... | ... | ... | ... | ... |

The header should contain unique sample names and the total number of taxonomically annotated reads. Taxonomic lineages are {rank}.{taxon} joined by a semicolon. Counts are absolute read counts. The higher rank lineage counts such as those of Bacteria are the sum of all bacterial daughter lineages and possible bacterial reads that could not be annotated at a lower rank. For pairwise comparissons between microbiomes at a specific rank (default: order), only the lineages at that rank are considered. The script does not check if the profiles are correct, for example if the number of reads associated with "super kingdom.Bacteria" is equal to or larger than the sum of reads associated with bacterial daughter taxa.

Alternatively, a relative abundance table can be supplied, in which case the header should only contain unique sample names.

| taxonomic lineage | MGYA00142528 | MGYA00142531 | MGYA00142543 | ... |
| --- | --- | --- | --- | --- |
| super kingdom.Archaea | 0 | 0 | 0 | ... |
| super kingdom.Bacteria | 1.0 | 1.0 | 1.0 | ... |
| super kingdom.Viruses | 0 | 0 | 0 | ... |
| super kingdom.Archaea;phylum.Candidatus_Aenigmarchaeota | 0 | 0 | 0 | ... |
| ... | ... | ... | ... | ... |

If a relative abundance table is supplied instead of a read count table, the `--c2 / --pairwise_comparisson_cutoff` (see below) can not be set to absolute read counts.

Let us know if other input formats would be useful.

The output file looks like this:

| taxonomic lineage | number of samples | mean relative abundance | SNB score |
| --- | --- | --- | --- |
| super kingdom.Archaea | 11256| 0.0959699 | 0.5215317 |
| super kingdom.Bacteria | 22295 | 0.9615460 | 0.5627098 |
| super kingdom.Viruses | 0 | nan | nan |
| super kingdom.Archaea;phylum.Candidatus_Aenigmarchaeota | 23 | 0.0013518 | 0.5174395 |
| ... | ... | ... | ... |

# Installation
Just download the script and you are good to go. Python >= 3.6 is required with the Python Standard Library, NumPy, and SciPy.

# Notes

* Alternatively to supplying a single file containing all taxonomic profiles (with the `-f / --microbiomes_file` option), a directory with multiple taxonomic profiles can be supplied (with the `-d / microbiomes_dir` option). 
* The `-d / --microbiomes_dir` option considers all files in the directory by default. You can consider just the subet with a certain suffix in the filename with the `-s / --suffix` option, like `.txt` or `.selection.txt`.
* A taxonomic profile in the `-d / --microbiomes_dir` directory does not need to contain taxonomic lineages that have an abundance of zero. In the example table above, only `super kingdom.Bacteria` needs to be present in the three depicted samples.
* Input file(s) can be plain text or .gzipped.
* You can change the relative abundance cut-off that defines whether a taxonomic lineage is considered present in a microbiome with the `--c1 / --presence_cutoff` [default: 1e-4].
* You can change the rank of pairwise comparisson with `-r / --rank_of_pairwise_comparisson` [default: order].
* You can change the abundance cut-off that defines whether a lineage is used for pairwise dissimilarity calculations of the taxonomic profiles with `--c2 / --pairwise_comparisson_cutoff`. This cut-off is independent of the `--c1 / --presence_cutoff`. Setting this option >= 1 assumes an absolute abundance cut-off (as used in the paper), if it is set < 1 a relative abundance cut-off is assumed [default: 5 reads].
* Both `--c1 / --presence_cutoff` and `--c2 / --pairwise_comparisson_cutoff` can be set to 0 to consider all taxonomic lineages with a non-zero abundance.
* The dissimilarity matrix can be written to an output file with the `-m / --write_dissimilarity_matrix` flag. Note that this option may make the script run considerably longer because the pairwise dissimilarities of all versus all samples have to be calculated instead of only the subset of combinations in which taxonomic lineages are found.
* It takes a couple of days to run the script on `table.MGnify.taxa_in_analyses.txt.gz`. Since the pairwise calculations scale roughly exponentially with sample size, smaller datasets will be run much quicker.
* Technical note: SNB scores generated by this script may differ slightly from those generated by the frozen code that was used for the paper due to differences in rounding of the pairwise distances (at ~1e-7). This script has a higher float precision. Output of the script can be made similar to that of the frozen code by adding `rho = float('{0:.7f}'.format(rho))` to the `get_pairwise_combinations()` function.
