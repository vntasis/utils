#!/usr/bin/awk -f

# Example script for using custom command line options
# Usage: awk -f options.awk -- -v -q file1 file2

BEGIN {
	for (i = 1; i < ARGC; i++) {
		if (ARGV[i] == "-v")
			verbose = 1
		else if (ARGV[i] == "-q")
			debug = 1
		else if (ARGV[i] ~ /^-./) {
			e = sprintf("%s: unrecognized option -- %c",
				ARGV[0], substr(ARGV[i], 2, 1))
			print e > "/dev/stderr"
		} else break
		delete ARGV[i]
	}
}
