<h1 align="center" id="title">Electrochemical Digital Twin</h1>

<p id="description">Electrochemical Digital Twin</p>

<h2>🛠️ Installation Steps:</h2>

<p>1. Clone the repo:

```bash
git clone git@github.com:infochemistry-ai/electrochem-digital-twins.git
```
<p>2. Create a virtual environment and install the necessary libraries </p>

  ```bash
  uv sync
  source .venv/bin/activate
  ```

<p>3. Download data from S3 bucket</p>
	
- Create a file .env according to the following template and insert the access keys
  ```bash
	S3_ACCESS_KEY=
	S3_SECRET_KEY=
  ```

- To download data from S3 bucket, run
  ```bash
	python download_data.py
  ```
