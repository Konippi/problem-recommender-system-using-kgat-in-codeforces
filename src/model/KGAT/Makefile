.PHONY: train
train-sm:
	@echo "Training the KGAT..."
	@rye run python main.py --sm

train:
	@echo "Training the KGAT..."
	@rye run python main.py

.PHONY: predict
predict-sm:
	@echo "Predicting the KGAT..."
	@rye run python main.py --sm --predict

predict:
	@echo "Predicting the KGAT..."
	@rye run python main.py --predict

.PHONY: visualize
# Need argument: dataset=training or dataset=test or dataset=validation
visualize-dataset-sm:
	@rye run python main.py --sm --visualize_dataset=$(dataset)

visualize-kg-sm:
	@rye run python main.py --sm --visualize_kg

visualize-attention-sm:
	@rye run python main.py --sm --visualize_attention

.PHONY: recommend
recommend-sm:
	@rye run python main.py --sm --recommend

