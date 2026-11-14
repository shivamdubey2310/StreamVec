.PHONY: check-env up down ps logs psql test clean

check-env:
	@echo "Checking Python, Docker, and directory scaffold..."
	@python3 --version
	@docker --version
	@docker compose version
	@test -f .env || (echo "Missing .env file! Run: cp .env.example .env" && exit 1)
	@for dir in producer worker api airflow/dags sql tests scripts archive; do \
		test -d $$dir || (echo "Missing directory: $$dir" && exit 1); \
	done
	@echo "Environment check passed."

up:
	docker compose up -d

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f --tail=100

psql:
	docker compose exec postgres psql -U streamvec -d streamvec

test:
	pytest -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
