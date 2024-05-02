# Prompts

The 10K nontoxic prompts we sample from [RealToxicityPrompts](https://www.aclweb.org/anthology/2020.findings-emnlp.301.pdf) is in `nontoxic_prompts-10k.jsonl`.

For sentiment experiments, we collect a new dataset of 10K prompts in `sentiment_prompts-10k/`. These prompts are half-sentences collected from OpenWebText (details in Appendix B of our paper). They are divided into 5K neutral prompts, 2.5K positive prompts, and 2.5K negative prompts, based on the base model's generations.

We have also incuded `toy_prompt.jsonl`, which contains a single prompt and demonstrates the format of the prompts dataset.