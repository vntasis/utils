#!/usr/bin/awk -f

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
