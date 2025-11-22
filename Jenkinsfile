pipeline {
    agent any

    environment {
        // CHANGE THIS TO YOUR PORT (example: 8001, 8002, etc.)
        STUDENT_PORT = '8140'
        DOCKER_IMAGE = 'my-gpu-service'
    }

    stages {
        stage('GPU Sanity Test') {
            steps {
                echo 'Installing required dependencies for cuda_test'
                sh 'nvidia-smi || echo "nvidia-smi not available"'
                echo 'Running CUDA sanity check...'
                sh '''
                    docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi
                    echo "✅ GPU access test passed"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                sh "docker build -t ${DOCKER_IMAGE}:latest ."
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                sh """
                    # Stop and remove old container if exists
                    docker stop ${DOCKER_IMAGE} || true
                    docker rm ${DOCKER_IMAGE} || true
                    
                    # Run new container with GPU access and your port
                    docker run -d \
                        --name ${DOCKER_IMAGE} \
                        --gpus all \
                        -p ${STUDENT_PORT}:${STUDENT_PORT} \
                        ${DOCKER_IMAGE}:latest
                    
                    echo "✅ Container deployed on port ${STUDENT_PORT}"
                """
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment completed successfully!"
            echo "📊 Your service is running on: http://10.90.90.100:${STUDENT_PORT}"
        }
        failure {
            echo "💥 Deployment failed. Check logs for errors."
        }
        always {
            echo "🧾 Pipeline finished."
        }
    }
}