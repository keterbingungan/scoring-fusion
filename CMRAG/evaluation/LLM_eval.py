import argparse

from utils.utils import load_json, save_json
from evaluation.LLM_judge import LLMJudge


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path', 
        type=str, 
        default='Qwen/Qwen2.5-7B-Instruct',
        help='The name or path of the pre-trained model.'
    )
    parser.add_argument(
        '--save_file',
        type=str,
        default='results_hotpotqa_num7404_top2_Qwen2.5-7B-Instruct.json',
        help='The name of the input file.'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='Number of samples to use in evaluation. -1 means all'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Number of queries per batch (used when using vLLM to generate multiple responses).'
    )
    parser.add_argument(
        '--use_vLLM',
        type=bool,
        default=True,
        help='Whether to use vLLM'
    )
    args = parser.parse_args()
    return args



def main():
    args = setup_args()
    
    # Load the prediction results
    pred_data = load_json(args.save_file)
    if args.num_samples > 0:
        pred_data = pred_data[:args.num_samples]
    print(f"Loaded {len(pred_data)} samples from {args.save_file}")

    # Initialize the LLM judge
    judge = LLMJudge(
        model_name_or_path=args.model_name_or_path,
        max_new_tokens=1024,
        USE_VLLM=args.use_vLLM,
        GPUID=0
    )

    # Evaluate each prediction
    results = []
    num_correct = 0
    # Use LLM to evaluate
    batch_size = args.batch_size
    for i in range(0, len(pred_data), batch_size):
        batch = pred_data[i:i+batch_size]
        qs = [item['query'] for item in batch]
        gold_anss = [item['answer'] for item in batch]
        gen_anss = [item['response'] for item in batch]
        # Filter <answer> and </answer> tags in the generated answers
        gen_anss = [ans.split("<answer>")[-1].split("</answer>")[0].strip() for ans in gen_anss]
        if args.use_vLLM:
            batch_results, batch_responses = judge.generate_response_batch(qs, gold_anss, gen_anss)
        else:
            batch_results, batch_responses = [], []
            for q, gold_ans, gen_ans in zip(qs, gold_anss, gen_anss):
                res, response = judge.generate_response(q, gold_ans, gen_ans)
                batch_results.append(res)
                batch_responses.append(response)
        
        # Store the results
        for item, res, response in zip(batch, batch_results, batch_responses):
            results.append({
                'query': item['query'],
                'gold_answer': item['answer'],
                'generated_answer': item['response'],
                'judge_result': res,
                'judge_response': response
            })
            
            if res:
                num_correct += 1
            
            if len(results) < 5:
                print(results[-1])
            
            # Show correctness rate every 100 samples
            if len(results) % 100 == 0:
                print(f"Processed {len(results)} samples, current accuracy: {num_correct / len(results):.4f}")
    
    print("="*50)
    print(f"Final accuracy: {num_correct / len(results):.4f}")
    print("="*50)

    # Save the evaluation results
    save_path = args.save_file.replace('.json', '_judged.json')
    save_json(results, save_path, type="json", use_indent=True)
    print(f"Saved evaluation results to {save_path}")



if __name__ == "__main__":
    main()