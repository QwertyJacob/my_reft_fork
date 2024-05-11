python train_defensive_intervention.py \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--layers "18;28" \
--low_rank_dimension 2 \
--n_train_examples 100 \
--batch_size 10 \
--learning_rate 4e-3 \
--num_train_epochs 5.0 \
--output_dir defense_results \
--logging_steps 1 \
--positions "f1+l1" \
--share_weights \
--nonstop
