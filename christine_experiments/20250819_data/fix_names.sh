for file in sonnet37_hack_*.jsonl; do
    base="${file%.jsonl}"
    prefix="${base%_*_*}"
    suffix="${base##*_}"
    middle="${base%_*}"
    middle="${middle##*_}"
    mv "$file" "${prefix}_${suffix}_${middle}.jsonl"
done
