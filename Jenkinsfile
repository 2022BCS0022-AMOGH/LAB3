pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python scripts/train.py
                '''
            }
        }

        stage('Read Metrics') {
            steps {
                sh '''
                . venv/bin/activate
                echo "===== MODEL METRICS ====="
                echo "2022BCS0022 - Amogh"

                python -c "
import json
m=json.load(open('metrics.json'))
print('R2:',m['r2'])
print('MSE:',m['mse'])
"
                '''
            }
        }

        stage('Docker Build') {
            steps {
                sh '''
                docker build -t amogh1029/wine-inference:latest .
                '''
            }
        }

        stage('Docker Push') {
            steps {
                withCredentials([
                    string(credentialsId: 'DOCKER_TOKEN', variable: 'DOCKER_TOKEN'),
                    string(credentialsId: 'DOCKER_USERNAME', variable: 'DOCKER_USERNAME')
                ]) {
                    sh '''
                    echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin
                    docker push amogh1029/wine-inference:latest
                    '''
                }
            }
        }
    }
}
