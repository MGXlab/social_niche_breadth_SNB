#!/usr/bin/env python3

__author__ = 'F. A. Bastiaan von Meijenfeldt'
__version__ = '0.2'
__date__ = '13 April, 2023'

import argparse
import copy
import gzip
import os
import numpy as np
import random
import scipy.stats as stats
import sys


def parse_arguments():
    class PathAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            path = os.path.expanduser(values.rstrip('/'))

            if not path.startswith('/') and not path.startswith('.'):
                path = f'./{path}'

            setattr(namespace, self.dest, path)

    parser = argparse.ArgumentParser(
            prog='calculate_SNB.py',
            description=('This script calculates the soial niche breadth '
                'score of all lineages in a set of microbiomes.'),
            usage=('./calculate_SNB.py (-f FILE | -d DIR) -o OUT_FILE '
                '[options] [-h / --help]'),
            add_help=False
            )

    required_choice = parser.add_argument_group('Required choice')
    group = required_choice.add_mutually_exclusive_group(required=True)
    group.add_argument(
            '-f',
            '--microbiomes_file',
            dest='microbiomes_file',
            metavar='',
            type=str,
            action=PathAction,
            help='[FILE] Path to file containing taxonomic profiles.'
            )
    group.add_argument(
            '-d',
            '--microbiomes_dir',
            dest='microbiomes_dir',
            metavar='',
            type=str,
            action=PathAction,
            help=(
                '[DIR] '
                'Path to directory containing taxonomic profiles. All files '
                'in the directory are considered by default. Alternatively, '
                'you can set the -s / --suffix option to only consider '
                'specific files.'
                )
            )

    required = parser.add_argument_group('Required')
    required.add_argument(
            '-o',
            '--output_file',
            dest='output_file',
            metavar='',
            required=True,
            type=str,
            action=PathAction,
            help=('[FILE] Path to output file. If you want to overwrite an '
                'existing file, use the --force flag.')
            )

    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument(
            '-s',
            '--suffix',
            dest='suffix',
            metavar='',
            required=False,
            type=str,
            help=('[STR] Suffix of taxonomic profiles (like .txt) when '
                '-d / --microbiomes_dir is set.')
            )
    optional.add_argument(
            '--c1',
            '--presence_cutoff',
            dest='presence_cutoff',
            metavar='',
            required=False,
            type=float,
            default=1e-4,
            help=(
                '[FLOAT] '
                'Relative abundance cut-off that defines whether a '
                'taxonomic lineage is considered present in a '
                'microbiome [default: 1e-4].'
                )
            )
    optional.add_argument(
            '-r',
            '--rank_of_pairwise_comparisson',
            dest='rank_of_pairwise_comparisson',
            metavar='',
            required=False,
            type=str,
            choices=['super_kingdom', 'phylum', 'class', 'order', 'family',
                'genus', 'species'],
            default='order',
            help=('[STR] Rank of comparisson for the pairwise dissimilarity '
                'calculations of the taxonomic profiles [default: order].')
            )
    optional.add_argument(
            '--c2',
            '--pairwise_comparisson_cutoff',
            dest='pairwise_comparisson_cutoff',
            metavar='',
            required=False,
            type=float,
            default=5.0,
            help=(
                '[INT | FLOAT] '
                'Abundance cut-off that defines whether a lineage is used '
                'for pairwise dissimilarity calculations of the taxonomic '
                'profiles at the taxonomic rank of comparisson. Setting this '
                'option >= 1 assumes  an absolute abundance cut-off, if it is '
                'set < 1 a relative abundance cut-off is assumed '
                '[default: 5 reads].'
                )
            )
    optional.add_argument(
            '-m',
            '--write_dissimilarity_matrix',
            dest='write_dissimilarity_matrix',
            required=False,
            action='store_true',
            help=(
                'Write all versus all pairwise dissimilarity matrix to '
                '${output_file}.dissimilarity_matrix. Note that setting this '
                'option may make the script run considerably longer. If you '
                'want to overwrite an existing file, use the --force flag.'
                )
            )
    optional.add_argument(
            '--force',
            dest='force',
            required=False,
            action='store_true',
            help='Force overwrite existing file.'
            )
    optional.add_argument(
            '-v',
            '--version',
            action='version',
            version=(f'v{__version__} ({__date__}).'),
            help='Print version information and exit.'
            )
    optional.add_argument(
            '-h',
            '--help',
            action='help',
            help='Show this help message and exit.'
            )

    (args, extra_args) = parser.parse_known_args()
    if len(extra_args) > 1:
        sys.exit('error: to many arguments supplied:\n{0}'.format(
            '\n'.join(extra_args)))

    if os.path.isfile(args.output_file) and not args.force:
        sys.exit(f'error: {args.output_file} already exists. If you want to '
                'overwrite it, use the --force flag.')
    with open(args.output_file, 'w') as outf:
        pass

    if args.write_dissimilarity_matrix:
        setattr(
                args,
                'matrix_output_file',
                f'{args.output_file}.dissimilarity_matrix'
                )
        if os.path.isfile(args.matrix_output_file) and not args.force:
            sys.exit(f'error: {args.matrix_output_file} already exists. If '
                    'you want to overwrite it, use the --force flag.')
        with open(args.matrix_output_file, 'w') as outf:
            pass

    if args.microbiomes_file and not os.path.isfile(args.microbiomes_file):
        sys.exit(f'error: {args.microbiomes_file} not found.')

    if args.microbiomes_dir and not os.path.isdir(args.microbiomes_dir):
        sys.exit(f'error: directory {args.microbiomes_dir} not found.')

    if not args.microbiomes_dir and args.suffix:
        print(
                '\n'
                '#######\n'
                'warning: -s / --suffix is set but this option is not used '
                'when a single file is supplied. If you mean to supply a '
                'directory instead, you can use the '
                '-d / -- microbiome_dir option.\n'
                '#######'
                )

    if args.pairwise_comparisson_cutoff >= 1:
        if not args.pairwise_comparisson_cutoff.is_integer():
            sys.exit(
                    'error: if --c2 / --pairwise_comparisson_cutoff >= 1, it '
                    'can only be set to round numbers, as it represents an '
                    'absolute number of reads.'
                )

        setattr(
                args,
                'pairwise_comparisson_cutoff',
                round(args.pairwise_comparisson_cutoff)
                )

    if args.presence_cutoff == 0:
        print(
                '\n'
                '#######\n'
                'warning: --c1 / --presence_cutoff is set to zero. All taxa '
                'with a non-zero abundance in a microbiome are considered '
                'present.\n'
                '#######'
                )

        setattr(
                args,
                'presence_cutoff',
                round(args.presence_cutoff)
                )

    if args.pairwise_comparisson_cutoff == 0:
        print(
                '\n'
                '#######\n'
                'warning: --c2 / --pairwise_comparisson_cutoff is set to '
                'zero. All lineages at the taxonomic rank of comparisson with '
                'a non-zero abundance in a microbiome are used for pairwise '
                'dissimilarity calculations.\n'
                '#######'
                )

        setattr(
                args,
                'pairwise_comparisson_cutoff',
                round(args.pairwise_comparisson_cutoff)
                )

    print()
    print(args)

    return args


def import_taxonomic_profiles(args):
    def import_single_file(file_, print_line_n=False):
        def parse_header(header_line):
            sample2i = {}
            sample2n_reads = {}
            relative_abundance_table = False

            error = False
            for i, header in enumerate(header_line):
                if i == 0:
                    continue

                header_split = header.split(' ')
                if len(header_split) == 2:
                    try:
                        sample, n_reads = header_split
                        n_reads = int(n_reads.lstrip('(').rstrip(')'))
                    except ValueError:
                        error = True

                    if relative_abundance_table:
                        # Samples have different formats.
                        error = True
                elif len(header_split) == 1:
                    relative_abundance_table = True

                    sample = header
                    # Set total number of reads to 1 for a relative abundance
                    # tables.
                    n_reads = 1
                else:
                    error = True

                if error:
                    sys.exit(
                            f'error: {file_} does not have the required '
                            'header format. The header should look like '
                            'this for a table with read counts:\n'
                            'taxonomic lineage<TAB>'
                            'unique_sample_name_without_spaces<SPACE>'
                            '(<total number of prokaryotic reads>)<TAB>'
                            '...etc\n'
                            '\n'
                            'For a relative abundance table, the header '
                            'should look like this:\n'
                            'taxonomic lineage<TAB>'
                            'unique_sample_name_without_spaces<TAB>...etc\n'
                            )

                if sample in sample2n_reads:
                    sys.exit('error: samples in header of '
                            f'{file_} are not unique.')

                sample2i[sample] = i
                sample2n_reads[sample] = n_reads

            return (sample2i, sample2n_reads, relative_abundance_table)

        lineage2samples = {}
        sample2lineages_to_compare = {}

        compressed = False
        if file_.endswith('.gz'):
            compressed = True

            f = gzip.open(file_, 'rb')
        else:
            f = open(file_, 'r')

        for n, line in enumerate(f):
            if print_line_n:
                print(f'Parsing line {n + 1:,}.', end='\r')

            if compressed:
                line = line.decode('utf-8')
            line = line.rstrip('\n').split('\t')

            if n == 0:
                (
                        sample2i,
                        sample2n_reads,
                        relative_abundance_table
                        ) = parse_header(line)
            else:
                lineage = line[0]
                if lineage in lineage2samples:
                    sys.exit(f'error: {lineage} is present multiple times '
                            f'in {file_}.')

                lineage2samples[lineage] = {}
                for sample, i in sample2i.items():
                    count = float(line[i])

                    if not relative_abundance_table:
                        # It's a read count table.
                        if not count.is_integer():
                            sys.exit(
                                    'error: read counts should be round '
                                    f'numbers. Error arose with {lineage} in '
                                    f'{file_}: {line[i]}.'
                                    )

                        count = round(count)
                        relative_abundance = count / sample2n_reads[sample]
                    else:
                        # It's a relative abundance table.
                        if count > 1:
                            sys.exit(
                                    'error: relative abundances should '
                                    f'be <= 1. Error arose with {lineage} in '
                                    f'{file_}: {line[i]}.\n'
                                    'If your table contains read counts '
                                    'instead of relative abundances, the '
                                    'total number of taxonomically annotated '
                                    'reads should be present in the header. '
                                    'See README.md.'
                                    )

                        relative_abundance = count

                    if count == 0:
                        # Only include taxa with an abundance > 0.
                        continue

                    if relative_abundance >= args.presence_cutoff:
                        # Only consider a lineage present in the sample if it
                        # has a relative abundance of at least the
                        # relative_abundance_cutoff.
                        lineage2samples[lineage][sample] = count

                    sample2lineages_to_compare.setdefault(sample, {})

                    if lineage.split(';')[-1].split('.')[
                            0] == args.rank_of_pairwise_comparisson:
                        # The taxonomic lineage has the rank of comparission.
                        if (args.pairwise_comparisson_cutoff >= 1 and
                                count >= args.pairwise_comparisson_cutoff):
                            # Only consider lineages with an absolute abundance
                            # cut-off for the pairwise dissimilarity
                            # calculations between samples.
                            sample2lineages_to_compare[sample][lineage] = count
                        if (args.pairwise_comparisson_cutoff < 1 and
                                relative_abundance >= args.pairwise_comparisson_cutoff):
                            # Only consider lineages with a relative abundance
                            # cut-off for the pairwise dissimilarity
                            # calculations between samples.
                            sample2lineages_to_compare[sample][lineage] = count

        if print_line_n:
            print()
        f.close()

        return (
                lineage2samples,
                sample2lineages_to_compare,
                sample2n_reads,
                relative_abundance_table
                )

    print()

    if args.microbiomes_file:
        print(f'Importing taxonomic profiles from {args.microbiomes_file}.')

        l2s, s2ltc, s2n, ra = import_single_file(
                args.microbiomes_file, print_line_n=True)

    if args.microbiomes_dir:
        l2s = {}
        s2ltc = {}
        s2n = {}
        ra = False

        if args.suffix:
            files = [file_ for file_ in os.listdir(args.microbiomes_dir) if
                    file_.endswith(args.suffix)]
            if len(files) == 0:
                sys.exit(f'error: no files with suffix {args.suffix} found '
                        f'in {args.microbiomes_dir}.')
        else:
            files = os.listdir(args.microbiomes_dir)
            if len(files) == 0:
                sys.exit(f'error: no files found in {args.microbiomes_dir}.')

        print(f'Importing taxonomic profiles from {len(files):,} files in '
                f'{args.microbiomes_dir}.')
        for n, file_ in enumerate(files):
            path = f'{args.microbiomes_dir}/{file_}'

            print(f'Importing {path} ({n + 1:,}).', end='\r')

            (
                    lineage2samples,
                    sample2lineages,
                    sample2n_reads,
                    relative_abundance_table
                    ) = import_single_file(path)

            if len(set(s2n) & set(sample2n_reads)) > 0:
                sys.exit(
                        'error: sample(s) '
                        f'{list(set(s2n) & set(sample2n_reads))} in '
                        'multiple files.'
                        )

            if not relative_abundance_table and ra:
                sys.exit(
                        f'error: {args.microbiomes_dir} contains both read '
                        'count tables and relative abundance tables. Error '
                        f'arose with {path}.'
                        )

            if relative_abundance_table:
                ra = True

            s2n = {**s2n, **sample2n_reads}

            for lineage in lineage2samples:
                l2s.setdefault(lineage, {})

                for sample in lineage2samples[lineage]:
                    l2s[lineage][sample] = lineage2samples[lineage][sample]

            for sample in sample2lineages:
                s2ltc[sample] = copy.deepcopy(sample2lineages[sample])
        print()

    if not ra:
        # It's a read count table.
        print(f'{len(s2n):,} taxonomic profiles with read counts imported '
                f'containing {len(l2s):,} taxonomic lineages.')
    else:
        # It's a relative abundance table.
        print(f'{len(s2n):,} taxonomic profiles with relative abundances '
                f'imported containing {len(l2s):,} taxonomic lineages.')

    return (l2s, s2ltc, s2n, ra)


def preflight_checks(l2s, s2ltc, s2n, ra, args):
    print()
    print('Doing some pre-flight checks.')

    if ra and args.pairwise_comparisson_cutoff >= 1:
        sys.exit(
                'error: the input file(s) contains relative abundances, which '
                'does not work with --c2 / --pairwise_comparisson_cutoff '
                f'set >= 1 (it is {args.pairwise_comparisson_cutoff} reads). '
                'You can either supply input file(s) that contain read count '
                'or set --c2 / --pairwise_comparisson_cutoff < 1. '
                'See README.md.'
                )

    warning1 = set()
    warning2 = set()
    warning3 = set()
    warning4 = set()
    for n, (lineage, samples) in enumerate(l2s.items()):
        print(
                'Checking taxonomic lineages '
                f'({(n + 1) / len(l2s) * 100:.2f})%.',
                end='\r')

        if len(samples) <= 1:
            warning1.add(lineage)
    print()

    for n, sample in enumerate(s2n):
        print(f'Checking {sample} ({(n + 1) / len(s2n) * 100:.2f}%).',
                end='\r')

        if not ra:
            # It's a read count table.
            if (args.presence_cutoff != 0 and
                    1 / s2n[sample] > args.presence_cutoff):
                warning2.add(sample)

            if (args.pairwise_comparisson_cutoff < 1 and
                    args.pairwise_comparisson_cutoff != 0 and
                    1 / s2n[sample] > args.pairwise_comparisson_cutoff):
                warning3.add(sample)

        if len(s2ltc[sample]) == 0:
            warning4.add(sample)
    print()

    if len(warning1) > 1:
        # Warning1 is not an error but a warning.
        print(
                '\n'
                '#######\n'
                f'warning: {len(warning1):,} taxonomic lineages are not '
                'present in two or more samples with the chosen '
                f'--c1 / --presence_cutoff ({args.presence_cutoff}). They are '
                'written to the output file.\n'
                '#######'
                )
    if len(warning2) > 0:
        # Warning2 is not an error but a warning for now.
        print(
                '\n'
                '#######\n'
                'warning: --c1 / --presence_cutoff ({0}) is set lower than the '
                'relative abundance of a single read in some samples. '
                'Consider excluding these samples or increasing the relative '
                'abundance cut-off.\n'
                'samples:\n'
                '\t{1}\n'
                '#######'.format(
                    args.presence_cutoff,
                    '\n\t'.join([f'{sample} (1/{s2n[sample]:,} reads)' for
                        sample in sorted(warning2)])
                    )
                )
    if len(warning3) > 0:
        # Warning3 is not an error but a warning for now.
        print(
                '\n'
                '#######\n'
                'warning: --c2 / --pairwise_comparisson_cutoff ({0}) is set '
                'lower than the relative abundance of a single read in some '
                'samples. Consider excluding these samples or increasing the '
                'relative abundance cut-off.\n'
                'samples:\n'
                '\t{1}\n'
                '#######'.format(
                    args.pairwise_comparisson_cutoff,
                    '\n\t'.join([f'{sample} (1/{s2n[sample]:,} reads)' for
                        sample in sorted(warning3)])
                    )
                )
    if len(warning4) > 0:
        # Warning4 is an error for now. I don't want to allow for this because
        # it generates confusion if some samples are not included in
        # the calculations.
        sys.exit(
                'error: some samples contain no lineages at rank {0} with at '
                'least {1} reads. These samples are ignored for SNB '
                'calculations. They should be removed from the dataset. '
                'Alternatively, you can change the rank of comparisson with '
                'the -r / --rank_of_pairwise_comparisson option, or decrease '
                'the --c2 / --pairwise_comparisson_cutoff.\n'
                'samples:\n'
                '\t{2}'.format(
                    args.rank_of_pairwise_comparisson,
                    args.pairwise_comparisson_cutoff,
                    '\n\t'.join(sorted(warning4))
                    )
                )

    if len(warning1 | warning2 | warning3 | warning4) == 0:
        print('Pre-flight checks done. Everything looks OK!')
    else:
        print()
        print('Pre-flight checks done. There are warnings (see above), but '
                'other than that everything is good to go!')

    return


def get_pairwise_combinations(l2s, args, s2n):
    """This function finds all pairwise combinations of samples that need to be
    compared. Depending on the distribution of taxonomic lineages in the
    microbiomes, this is a subset of all versus all.
    """
    print()
    print(f'Finding all combinations of samples for pairwise comparissons.')

    pairwise_combinations = set()

    # Find all sample combinations.
    list_of_sample_sets = []
    for lineage in l2s:
        sample_set = set(l2s[lineage])
        # Include each sample set only once.
        if sample_set in list_of_sample_sets:
            continue
        list_of_sample_sets.append(sample_set)

    if args.write_dissimilarity_matrix:
        # Find all versus all pairwise combinations if the dissimilarity
        # matrix is written.
        sample1_trace = set()
        for sample1 in s2n:
            for sample2 in s2n:
                if sample1 == sample2:
                    continue

                if sample2 in sample1_trace:
                    continue

                pairwise_combinations.add((sample1, sample2))
            sample1_trace.add(sample1)
    else:
        # If the dissimilarity matrix is not written, only find those
        # combinations of samples in which taxonomic lineages are present.

        # Reduce redundancy further.
        no_subsets = []
        for sample_set1 in sorted(
                list_of_sample_sets, key=lambda x: len(x), reverse=True):
            for sample_set2 in no_subsets:
                if sample_set1.issubset(sample_set2):
                    break
            else:
                no_subsets.append(sample_set1)

        # Get all pairwise combinations.
        for sample_set in no_subsets:
            sample1_trace = set()
            for sample1 in sample_set:
                for sample2 in sample_set:
                    if sample1 == sample2:
                        continue

                    if sample2 in sample1_trace:
                        continue

                    pairwise_combinations.add((sample1, sample2))
                sample1_trace.add(sample1)

    return (pairwise_combinations, list_of_sample_sets)


def calculate_pairwise_dissimilarities(pairwise_combinations, s2ltc):
    print()
    print(f'Calculating {len(pairwise_combinations):,} '
            'pairwise dissimilarities.')

    pairwise_dissimilarity = {}

    for n, (sample1, sample2) in enumerate(pairwise_combinations):
        if n % 100 == 0:
            print(f'Working on pairwise dissimilarity {n:,} '
                    f'({n / len(pairwise_combinations) * 100:.2f}%).',
                    end='\r')

        lineages_to_compare_1 = set(s2ltc[sample1])
        lineages_to_compare_2 = set(s2ltc[sample2])

        union = lineages_to_compare_1 | lineages_to_compare_2
        if len(union) == 1:
            # Set dissimilarity to zero if there is only 1 lineage in
            # the union.
            pairwise_dissimilarity[(sample1, sample2)] = 0
        else:
            sorted_union = sorted(union)

            # Fill zeros for those taxa that do not appear in the other
            # microbiome.
            l1 = [s2ltc[sample1][lineage] if lineage in 
                    lineages_to_compare_1 else 0 for lineage in sorted_union]
            l2 = [s2ltc[sample2][lineage] if lineage in
                    lineages_to_compare_2 else 0 for lineage in sorted_union]

            rho, p = stats.spearmanr(l1, l2)
            dissimilarity = 0.5 - rho / 2

            pairwise_dissimilarity[(sample1, sample2)] = dissimilarity
    print('Pairwise dissimilarity calculations done.', ' ' * 30)

    return pairwise_dissimilarity


def write_dissimilarity_matrix(pairwise_dissimilarity, args):
    print()
    print('Writing all versus all pairwise dissimilarities '
            f'to {args.matrix_output_file}.')

    with open(args.matrix_output_file, 'w') as outf:
        for (
                (sample1, sample2), dissimilarity
                ) in pairwise_dissimilarity.items():
            outf.write(f'{sample1}\t{sample2}\t{dissimilarity}\n')
    print(f'{args.matrix_output_file} written.')

    return


def calculate_SNB(list_of_sample_sets, pairwise_dissimilarity, l2s):
    print()
    print(f'Calculating SNB score for {len(l2s):,} taxonomic lineages.')

    SNB = {}

    # Only calculate once for each sample set.
    SNB_sample_sets = {}
    for n, sample_set in enumerate(
            random.sample(list_of_sample_sets, len(list_of_sample_sets))):
        # I randomise a bit to make the progress indicator more accurate.
        print(f'{(n + 1) / len(list_of_sample_sets) * 100:.2f}%.', end='\r')

        if len(sample_set) <= 1:
            continue

        dissimilarities = []

        sample1_trace = set()
        for sample1 in sample_set:
            for sample2 in sample_set:
                if sample1 == sample2:
                    continue

                if sample2 in sample1_trace:
                    continue

                try:
                    dissimilarities.append(pairwise_dissimilarity[
                        (sample1, sample2)])
                except:
                    dissimilarities.append(pairwise_dissimilarity[
                        (sample2, sample1)])
            sample1_trace.add(sample1)

        key = tuple(sorted(sample_set))
        SNB_sample_sets[key] = np.mean(dissimilarities)

    for lineage, samples in l2s.items():
        if len(samples) <= 1:
            SNB[lineage] = np.nan

            continue

        key = tuple(sorted(samples))

        SNB[lineage] = SNB_sample_sets[key]

    print(f'Done calculating SNB score for {len(SNB):,} taxonomic lineages.')

    return SNB


def write_file(SNB, l2s, s2n, args):
    print()
    print(f'Writing SNB scores to {args.output_file}.')

    with open(args.output_file, 'w') as outf:
        outf.write(
                'taxonomic lineage\t'
                'number of samples\t'
                'mean relative abundance\t'
                'SNB score\n'
                )

        for lineage in sorted(l2s, key=lambda x: (len(x.split(';')), x)):
            if len(l2s[lineage]) == 0:
                mean_relative_abundance = np.nan
            else:
                mean_relative_abundance = np.mean(
                        [l2s[lineage][sample] / s2n[sample] for
                            sample in l2s[lineage]])

            outf.write(
                    f'{lineage}\t'
                    f'{len(l2s[lineage])}\t'
                    f'{mean_relative_abundance:.7f}\t'
                    f'{SNB[lineage]:.7f}\n'
                    )
    print(f'{args.output_file} written.')

    return


def main():
    args = parse_arguments()

    l2s, s2ltc, s2n, ra = import_taxonomic_profiles(args)

    # Do some basic checks on the samples.
    preflight_checks(l2s, s2ltc, s2n, ra, args)

    # Calculate pairwise dissimilarities.
    (
            pairwise_combinations, list_of_sample_sets
            ) = get_pairwise_combinations(l2s, args, s2n)
    pairwise_dissimilarity = calculate_pairwise_dissimilarities(
            pairwise_combinations, s2ltc)

    if args.write_dissimilarity_matrix:
        write_dissimilarity_matrix(pairwise_dissimilarity, args)

    # Calculate SNB.
    SNB = calculate_SNB(list_of_sample_sets, pairwise_dissimilarity, l2s)

    # Write output file.
    write_file(SNB, l2s, s2n, args)

    print()
    print('Done!:)')


if __name__ == '__main__':
    main()
