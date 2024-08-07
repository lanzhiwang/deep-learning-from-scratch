```bash

python -m venv .env

source .env/bin/activate

docker run -ti --rm \
-v ~/work/code/py_code/deep-learning/01-deep-learning-from-scratch:/01-deep-learning-from-scratch \
-w /01-deep-learning-from-scratch \
docker-mirrors.alauda.cn/library/python:3.10.12-bullseye \
bash

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple yapf

find . -name "*.py" -exec yapf -i {} \;


```
