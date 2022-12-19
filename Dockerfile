## lets specify the base image
FROM python:3.10-slim

## sets a special python settings for being able to see logs 
ENV PYTHONUNBUFFERED=TRUE

## Installs Pipenv 
RUN pip --no-cache-dir install pipenv 

## sets the working directory to /app 
WORKDIR /app

## copies our pipenv files 
COPY ["Pipfile", "Pipfile.lock", "./"]

## installs the dependencies from the pipenv files 
RUN pipenv install --deploy --system && rm -rf /root/.cache

## copies our code as well as the model 
COPY ["*.py", "generator_failure_prediction_model.pkl", "./"]

## opens the port that our web services uses 
EXPOSE 9696 

## lets specify how the service should be started
ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "predict:app" ]
