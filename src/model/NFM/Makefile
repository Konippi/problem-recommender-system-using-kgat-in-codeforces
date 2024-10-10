.PHONY: train
train-sm:
	@echo "Training the NFM..."
	@rye run python main.py --sm

train:
	@echo "Training the NFM..."
	@rye run python main.py

.PHONY: predict
predict-sm:
	@echo "Predicting the NFM..."
	@rye run python main.py --sm --predict

predict:
	@echo "Predicting the NFM..."
	@rye run python main.py --predict
