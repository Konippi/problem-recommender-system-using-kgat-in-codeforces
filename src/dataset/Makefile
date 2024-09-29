SIZE := "all"

.PHONY: create-user-and-submission
create-user-and-submission:
	@echo "Creating user and submission dataset..."
	@if [ $(SIZE) = "sm" ]; then \
		python create_user_and_submission_dataset.py --sm; \
	else \
		python create_user_and_submission_dataset.py; \
	fi

.PHONY: create-problem
create-problem:
	@echo "Creating problem dataset..."
	@python create_problem_dataset.py

.PHONY: create-relation
create-relation:
	@echo "Creating relation dataset..."
	@python create_relation_dataset.py

.PHONY: create
create:
	make create-problem
	@echo ""
	
	make create-user-and-submission
	@echo ""

	make create-relation
	@echo ""

.PHONY: create-sm
create-sm:
	make create-problem
	@echo ""

	make create-user-and-submission SIZE=sm
	@echo ""

	make create-relation
	@echo ""
	