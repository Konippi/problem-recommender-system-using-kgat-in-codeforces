.PHONY: install
install:
	@rye sync

.PHONY: train
train-sm:
	@make -C src/model/KGAT train-sm

train:
	@make -C src/model/KGAT train

.PHONY: predict
predict-sm:
	@make -C src/model/KGAT predict-sm

predict:
	@make -C src/model/KGAT predict

.PHONY: recommend
recommend-sm:
	@make -C src/model/KGAT recommend-sm

.PHONY: testing
testing-sm:
	@make -C src/model/KGAT testing-sm