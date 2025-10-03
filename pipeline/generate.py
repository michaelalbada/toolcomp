import os

from inference.react_inference import generate as react_generate
from inference.native_inference import generate as native_generate
from pipeline.utils import save_json
from model.utils import load_model
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenerationPipeline:

    def __init__(self, args):
        self.args = args

    def prepare_inference_func(self, input_data, args):

        policy_model = load_model(args.policy_sampling_params['model'], args.policy_generation_strategy, args.policy_sampling_params)

        inference_func = react_generate
        inference_args = {
            "input_data": input_data,
            "policy_model": policy_model,
            "num_retries": args.num_retries,
            "num_full_retries": args.num_full_retries,
            "max_depth": args.max_depth,
        }

        return inference_func, inference_args

    def save_data(self, react_trees):
        generations_file_path = os.path.join(self.args.output_dir, f"generations.json")
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_json(react_trees, generations_file_path)

    def iter_save_data(self, running_futures, react_trees, n_samples):
        """Process completed futures as they finish using as_completed()"""
        completed_count = 0
        failed_count = 0

        with tqdm(total=n_samples, desc="Processing") as pbar:
            for future in as_completed(running_futures):
                try:
                    # Get result with timeout to prevent hanging
                    generation, metadata = future.result(timeout=60)  # 1 min timeout per task

                    react_trees.append(generation)
                    completed_count += 1

                    # Update progress bar with stats
                    pbar.set_postfix({
                        'completed': completed_count,
                        'failed': failed_count
                    })
                    pbar.update(1)

                    # Save checkpoint every 10 completions
                    if completed_count % 10 == 0:
                        self.save_data(react_trees)
                        logger.info(f"Checkpoint saved: {completed_count}/{n_samples} completed")

                except TimeoutError as e:
                    failed_count += 1
                    logger.error(f"Task timed out after 600 seconds")
                    pbar.update(1)

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Task failed with error: {str(e)}")
                    pbar.update(1)

            self.save_data(react_trees)
            logger.info(f"Pipeline completed: {completed_count} succeeded, {failed_count} failed")

            return react_trees

    def generate(self, input_data, react_trees):
        n_samples = len(input_data)
        args = self.args

        logger.info(f"Starting generation for {n_samples} samples with {args.num_workers} workers")

        inference_func, inference_args = self.prepare_inference_func(input_data, args)

        # Use context manager to properly manage executor lifecycle
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:

            if self.args.tool_use_strategy == "react":
                logger.info("Using ReAct inference strategy")
                futures = [executor.submit(
                    inference_func,
                    [input_sample],
                    inference_args['policy_model'],
                    inference_args['num_retries'],
                    inference_args['num_full_retries'],
                    inference_args['max_depth'],
                    index) for index, input_sample in enumerate(input_data)]

            elif self.args.tool_use_strategy == "native":
                logger.info("Using native inference strategy")
                futures = [executor.submit(
                    native_generate,
                    [input_sample],
                    inference_args['policy_model'],
                    inference_args['num_full_retries'],
                    index, args.apply_chat_template
                    ) for index, input_sample in enumerate(input_data)]
            else:
                raise ValueError(f"Unsupported tool call format: {args.tool_call_format}")

            # Process futures as they complete
            react_trees = self.iter_save_data(futures, react_trees, n_samples)
