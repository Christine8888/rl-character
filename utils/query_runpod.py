import requests
import time
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv('/workspace/rl-character/safety-tooling/.env')

# Configuration
RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
GPU_TYPE = 'NVIDIA H200'
GPU_COUNT = 1
VOLUME_ID = '74u08td4sg'  # christine_gpu_volume in US-CA-2
DATACENTER_ID = 'US-CA-2'  # Datacenter where the network volume is located
CHECK_INTERVAL = 300  # Check every 5 minutes
NAME = 'christine-4xh200'
IMAGE_NAME = 'runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04'

# RunPod GraphQL API endpoint
API_URL = 'https://api.runpod.io/graphql'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {RUNPOD_API_KEY}'
}

def query_gpu_availability():
    """Query available GPUs matching our criteria in secure cloud and specific datacenter"""
    query = """
    query GpuTypes {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        securePrice
        lowestPrice(input: {
          gpuCount: %d
          secureCloud: true
          dataCenterId: "%s"
        }) {
          minimumBidPrice
          uninterruptablePrice
        }
      }
    }
    """ % (GPU_COUNT, DATACENTER_ID)

    response = requests.post(API_URL, json={'query': query}, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error querying GPU types: {response.status_code}")
        print(response.text)
        return None

def check_max_available_gpus(gpu_type_id):
    """Check the maximum number of GPUs available for a specific GPU type in the network volume's datacenter"""
    # Try to find available pods by querying with different GPU counts
    for count in range(GPU_COUNT, 0, -1):
        query = """
        query CheckAvailability {
          gpuTypes(input: {id: "%s"}) {
            id
            displayName
            lowestPrice(input: {
              gpuCount: %d
              secureCloud: true
              dataCenterId: "%s"
            }) {
              uninterruptablePrice
            }
          }
        }
        """ % (gpu_type_id, count, DATACENTER_ID)

        response = requests.post(API_URL, json={'query': query}, headers=headers)

        if response.status_code == 200:
            result = response.json()
            if result.get('data') and result['data'].get('gpuTypes'):
                gpu_data = result['data']['gpuTypes'][0]
                if gpu_data.get('lowestPrice') and gpu_data['lowestPrice'].get('uninterruptablePrice'):
                    return count

    return 0

def find_available_pods():
    """Find available pods with our GPU configuration"""
    query = """
    query Pods {
      myself {
        pods {
          id
          name
          runtime
          gpuCount
          machine {
            gpuDisplayName
          }
        }
      }
      podFindAndDeployOnDemand(input: {
        cloudType: SECURE
        gpuCount: %d
        networkVolumeId: "%s"
        containerDiskInGb: 100
        minVcpuCount: 8
        minMemoryInGb: 32
        gpuTypeId: "%s"
        name: "%s"
        dockerArgs: ""
        ports: "8888/http,22/tcp"
        volumeMountPath: "/workspace"
      }) {
        id
        costPerHr
        gpuCount
      }
    }
    """ % (GPU_COUNT, VOLUME_ID, GPU_TYPE, NAME, IMAGE_NAME)
    
    response = requests.post(API_URL, json={'query': query}, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error finding pods: {response.status_code}")
        print(response.text)
        return None

def rent_pod():
    """Attempt to rent a pod with 4xH200"""
    mutation = """
    mutation {
      podFindAndDeployOnDemand(
        input: {
          cloudType: SECURE
          gpuCount: %d
          networkVolumeId: "%s"
          containerDiskInGb: 50
          minVcpuCount: 16
          minMemoryInGb: 64
          gpuTypeId: "%s"
          name: "%s"
          imageName: "%s"
          dockerArgs: ""
          ports: "8888/http,22/tcp"
          volumeMountPath: "/workspace"
        }
      ) {
        id
        imageName
        costPerHr
        gpuCount
        machine {
          gpuDisplayName
        }
      }
    }
    """ % (GPU_COUNT, VOLUME_ID, GPU_TYPE, NAME, IMAGE_NAME)

    response = requests.post(API_URL, json={'query': mutation}, headers=headers)

    if response.status_code == 200:
        result = response.json()
        if 'errors' in result:
            return None, result['errors']
        return result, None
    else:
        error_msg = f"HTTP Error: {response.status_code}\nResponse: {response.text}"
        return None, error_msg

def main():
    print(f"Starting RunPod monitoring for {GPU_COUNT}x{GPU_TYPE}")
    print(f"Using volume: {VOLUME_ID} in datacenter: {DATACENTER_ID}")
    print(f"Checking every {CHECK_INTERVAL} seconds\n")

    while True:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking availability...")

        # Check GPU availability
        gpu_data = query_gpu_availability()

        if gpu_data and 'data' in gpu_data:
            h200_found = False
            for gpu in gpu_data['data']['gpuTypes']:
                if 'H200' in gpu['displayName']:
                    h200_found = True
                    print(f"Found: {gpu['displayName']}")

                    # Check if available in secure cloud (required for network volumes)
                    if not gpu.get('secureCloud'):
                        print(f"  Not available in Secure Cloud (required for network volumes)")
                        continue

                    # Check what the maximum available GPU count is
                    print(f"  Checking max available cluster size...")
                    max_available = check_max_available_gpus(gpu['id'])
                    print(f"  Max GPUs available: {max_available} (need {GPU_COUNT})")

                    if max_available >= GPU_COUNT:
                        # Check pricing
                        lowest_price = gpu.get('lowestPrice')
                        if lowest_price and lowest_price.get('uninterruptablePrice'):
                            price = lowest_price['uninterruptablePrice']
                            print(f"  Price: ${price}/hr for {GPU_COUNT} GPUs")
                            print("  Attempting to rent...")
                            result, error = rent_pod()

                            if result and 'data' in result:
                                pod = result['data']['podFindAndDeployOnDemand']
                                if pod:
                                    print(f"\n✓ SUCCESS! Pod rented:")
                                    print(f"  Pod ID: {pod['id']}")
                                    print(f"  GPU Count: {pod['gpuCount']}")
                                    print(f"  Cost: ${pod['costPerHr']}/hr")
                                    print(f"\nExiting script.")
                                    return

                            if error:
                                print(f"  Failed to rent: {error}")
                        else:
                            print(f"  Pricing unavailable")
                    else:
                        print(f"  Not enough GPUs available for {GPU_COUNT}x cluster")

            if not h200_found:
                print("No H200 GPUs found in availability data")

        print(f"Waiting {CHECK_INTERVAL} seconds before next check...\n")
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    # Validate API key is set
    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not found in /workspace/rl-character/safety-tooling/.env")
        exit(1)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
