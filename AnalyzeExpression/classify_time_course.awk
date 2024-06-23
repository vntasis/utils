#!/usr/bin/awk -f

# Classify time course data
#
# Display help message
# classify_time_course.awk -- -h
#
# Author: Vasilis Ntasis
# Adjusted Awk version of a Perl script written by Ramil Nurtdinov

function print_help() {
    help_message = "Classify time course data\n" \
                 "=========================\n" \
                 "Usage: classify_time_course.awk [Parameters] -- [-h] INPUT-FILE\n" \
                 "\n" \
                 "Parameters (can be specified using the -v option of awk):\n" \
                 "  comp_threshold              Comparison threshold (default: 0.25)\n" \
                 "  fold_change_cutoff          Fold change cutoff (default: 1)\n" \
                 "  high_expression_cutoff      High expression cutoff (default: 4)\n" \
                 "  low_expression_cutoff       Low expression cutoff (default: 1)\n" \
                 "  need_logarithm              Need logarithm (default: 0)\n" \
                 "  peak_height                 Peak height (default: 1)\n" \
                 "  prof_threshold              Profile threshold (default: 0.50)\n" \
                 "  summary_file                Summary file (default: classes_summary.tsv)\n" \
                 "  tc_length                   Time course length (default: 3)\n" \
                 "\n" \
                 "Options:\n" \
                 "  -h                          Display this help message and exit\n" \
                 "\n" \
                 "Input:\n" \
                 "  INPUT-FILE must be a tab-separated file of time course data\n" \
                 "  with a header in the first line. Each row represents a measured variable,\n" \
                 "  and each column represents a consecutive time-point.\n" \
                 "  The first column should contain a variable-specific identifier.\n" \
                 "  E.g.\n" \
                 "    id\ttp1\ttp2\ttp3\ttp4\n" \
                 "    g1\t5\t50\t100\t200\n" \
                 "    g3\t100\t50\t5\t1\n" \
                 "\n" \
                 "Output:\n" \
                 "  Classifies each variable in the input time course dataset\n" \
                 "  in on of the following profiles:\n" \
                 "    upregulation\n" \
                 "    downregulation\n" \
                 "    peaking\n" \
                 "    bending\n" \
                 "    high expression\n" \
                 "    moderate expression\n" \
                 "    low expression\n" \
                 "\n" \
                 "Examples:\n" \
                 "  ./classify_time_course.awk -v comp_threshold=0.3 -v fold_change_cutoff=2 -- input-file.tsv\n" \
                 "  ./classify_time_course.awk -- -h"

    print help_message
}

function log2(x) { return log(1+x)/log(2) }

function abs(x) { return x < 0 ? -x : x }

function give_class_up(up, down, class, tp) {
    if (up > down && down <= prof_threshold && (up - down) >= (1 - comp_threshold)) {
        print id, class, tp
        count[class]++
        next
    }
}

function give_class_down(up, down, class, tp) {
    if (down > up && up <= prof_threshold && (down - up) >= (1 - comp_threshold)) {
        print id, class, tp
        count[class]++
        next
    }
}

function get_up_down(last_tp, total_diff, out) {
    for (f=3; f<=(last_tp+1); f++) {
        diff = abs($f - $(f-1))
        $f >= $(f-1) ? out["up"] += diff : out["down"] += diff
    }

    out["up"] = out["up"] / total_diff
    out["down"] = out["down"] / total_diff
}

function get_peaking(max_value, max_index, last_tp, total_diff, out) {

    if (max_index < 2 || max_index > (last_tp - 1)) { return }

    diff_begin = max_value - $2
    diff_end = max_value - $(tc_length + 1)

    if (diff_begin < peak_height || diff_end < peak_height) { return }

    new_max = max_value
    for (f=3; f<=(last_tp+1); f++) {
        if (f < (max_index+2)) {
            tp1 = $(f-1)
            tp2 = $f
        } else if (f == (max_index+2)) {
            tp1 = $(f-1)
            tp2 = max_value + max_value - $f
            new_max = tp2 > new_max ? tp2 : new_max
        } else {
            tp1 = max_value + max_value - $(f-1)
            tp2 = max_value + max_value - $f
            new_max = tp1 > new_max ? tp1 : new_max
            new_max = tp2 > new_max ? tp2 : new_max
        }
        diff = abs(tp2 - tp1)
        tp2 >= tp1 ? out["up"] += diff : out["down"] += diff
    }

    new_total_diff = total_diff + (new_max - max_value)
    out["up"] = out["up"] / new_total_diff
    out["down"] = out["down"] / new_total_diff
}

function get_bending(min_value, min_index, last_tp, total_diff, out) {

    if (min_index < 2 || min_index > (last_tp - 1)) { return }

    diff_begin = $2 - min_value
    diff_end = $(tc_length + 1) - min_value

    if (diff_begin < peak_height || diff_end < peak_height) { return }

    new_min = min_value
    for (f=3; f<=(last_tp+1); f++) {
        if (f < (min_index+2)) {
            tp1 = $(f-1)
            tp2 = $f
        } else if (f == (min_index+2)) {
            tp1 = $(f-1)
            tp2 = min_value + min_value - $f
            new_min = tp2 < new_min ? tp2 : new_min
        } else {
            tp1 = min_value + min_value - $(f-1)
            tp2 = min_value + min_value - $f
            new_min = tp1 < new_min ? tp1 : new_min
            new_min = tp2 < new_min ? tp2 : new_min
        }
        diff = abs(tp2 - tp1)
        tp2 >= tp1 ? out["up"] += diff : out["down"] += diff
    }

    new_total_diff = total_diff + (min_value - new_min)
    out["up"] = out["up"] / new_total_diff
    out["down"] = out["down"] / new_total_diff
}

BEGIN {

    for (i = 1; i < ARGC; i++) {
        if (ARGV[i] == "-h") { print_help(); exit; }
        else if (ARGV[i] ~ /^-./) {
            e = sprintf("%s: unrecognized option -- %c",
                    ARGV[0], substr(ARGV[i], 2, 1))
            print e > "/dev/stderr"
            exit 2
        } else
            break
    }

    # Params
    comp_threshold          = comp_threshold == "" ? 0.25 : comp_threshold
    fold_change_cutoff      = fold_change_cutoff == "" ? 1 : fold_change_cutoff
    high_expression_cutoff  = high_expression_cutoff == "" ? 4 : high_expression_cutoff
    low_expression_cutoff   = low_expression_cutoff == "" ? 1 : low_expression_cutoff
    need_logarithm          = need_logarithm == "" ? 0 : need_logarithm
    peak_height             = peak_height == "" ? 1 : peak_height
    prof_threshold          = prof_threshold == "" ? 0.50 : prof_threshold
    summary_file            = summary_file == "" ? "classes_summary.tsv" : summary_file
    tc_length               = tc_length == "" ? 3 : tc_length

    FS=OFS="\t"
    print "ID", "Class", "Timepoint"
}

NR == 1 { for (f=1; f<=tc_length; f++) { tps[f] = $(f+1) } }

NR > 1 {
    id = $1

    if (need_logarithm) { for (f=2; f<=(tc_length+1); f++) { $f = log2($f) } }


    min = $2
    min_point = 1
    max = $2
    max_point = 1
    sum = $2

    for (f=3; f<=(tc_length+1); f++) {
        sum += $f
        min_point = $f < min ? (f - 1) : min_point
        min = $f < min ? $f : min
        max_point = $f > max ? (f - 1) : max_point
        max = $f > max ? $f : max
    }

    fc = max - min
    av = sum / tc_length

    if (av < low_expression_cutoff) {
        print id, "low expression", "flat"
        count["low expression"]++
        next
    }

    if (fc < fold_change_cutoff) {
        if (av < high_expression_cutoff) {
            print id, "moderate expression", "flat"
            count["moderate expression"]++
        } else {
            print id, "high expression", "flat"
            count["high expression"]++
        }
        next
    }

    up_down_reg["up"] = 0
    up_down_reg["down"] = 0
    get_up_down(tc_length, fc, up_down_reg)
    #print id, up_down_reg["up"], up_down_reg["down"]

    give_class_up(up_down_reg["up"], up_down_reg["down"], "upregulation", tps[max_point])
    give_class_down(up_down_reg["up"], up_down_reg["down"], "downregulation", tps[min_point])

    peaking["up"] = 0
    peaking["down"] = 0
    get_peaking(max, max_point, tc_length, fc, peaking)
    #print id, peaking["up"], peaking["down"]
    give_class_up(peaking["up"], peaking["down"], "peaking", tps[max_point])

    bending["up"] = 0
    bending["down"] = 0
    get_bending(min, min_point, tc_length, fc, bending)
    #print id, bending["up"], bending["down"]
    give_class_down(bending["up"], bending["down"], "bending", tps[min_point])

    print id, "variable", "variable"
    count["variable"]++

}

END {
    for (c in count) {
        print c, count[c] > summary_file
    }
}
