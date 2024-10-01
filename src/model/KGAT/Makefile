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
visualize-kg-sm:
	@rye run python main.py --sm --kg_visualize

.PHONY: recommend
recommend-sm:
	@rye run python main.py --sm --recommend

.PHONY: testing
testing-sm:
	@rye run python main.py --sm --testing
