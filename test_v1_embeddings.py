import requests
import json

def test_v1_embeddings_endpoint():
    """Test the /v1/embeddings endpoint for OpenAI compatibility"""
    url = "http://localhost:8080/v1/embeddings"
    
    # Test data
    test_cases = [
        {
            "name": "Single string input",
            "data": {
                "input": "Hello, how are you?",
                "model": "minishlab/potion-base-8M"
            }
        },
        {
            "name": "Array input",
            "data": {
                "input": ["Hello, how are you?", "I am fine, thank you"],
                "model": "minishlab/potion-base-8M"
            }
        },
        {
            "name": "With encoding format",
            "data": {
                "input": "Hello, how are you?",
                "model": "minishlab/potion-base-8M",
                "encoding_format": "float"
            }
        },
        {
            "name": "Invalid model test",
            "data": {
                "input": "Hello, how are you?",
                "model": "invalid-model"
            },
            "expect_error": True
        }
    ]
    
    try:
        for test_case in test_cases:
            print(f"\n--- Testing: {test_case['name']} ---")
            
            response = requests.post(url, json=test_case["data"])
            print(f"Status Code: {response.status_code}")
            
            if test_case.get("expect_error"):
                if response.status_code == 400:
                    print("✅ Expected error handled correctly")
                    print("Error:", response.json().get("error"))
                else:
                    print("❌ Expected error but got different status")
            else:
                if response.status_code == 200:
                    data = response.json()
                    print("Response structure:")
                    print(f"  - object: {data.get('object')}")
                    print(f"  - model: {data.get('model')}")
                    print(f"  - data length: {len(data.get('data', []))}")
                    print(f"  - usage: {data.get('usage')}")
                    
                    # Verify OpenAI compatibility
                    assert "object" in data and data["object"] == "list"
                    assert "data" in data and isinstance(data["data"], list)
                    assert "model" in data
                    assert "usage" in data
                    
                    if len(data["data"]) > 0:
                        first_embedding = data["data"][0]
                        assert "object" in first_embedding and first_embedding["object"] == "embedding"
                        assert "index" in first_embedding
                        assert "embedding" in first_embedding
                        print(f"  - embedding dimensions: {len(first_embedding['embedding'])}")
                    
                    print("✅ OpenAI compatibility verified!")
                else:
                    print(f"❌ Request failed with status {response.status_code}")
                    print("Response:", response.text)
                    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on localhost:8080")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_model_validation():
    """Test that model validation works properly"""
    print("\n=== Testing Model Validation ===")
    
    # First get available models
    models_url = "http://localhost:8080/v1/models"
    embeddings_url = "http://localhost:8080/v1/embeddings"
    
    try:
        models_response = requests.get(models_url)
        if models_response.status_code == 200:
            models_data = models_response.json()
            available_model = models_data["data"][0]["id"]
            print(f"Available model: {available_model}")
            
            # Test with correct model
            test_data = {
                "input": "Test input",
                "model": available_model
            }
            
            response = requests.post(embeddings_url, json=test_data)
            if response.status_code == 200:
                print("✅ Correct model accepted")
            else:
                print("❌ Correct model rejected")
                
            # Test with incorrect model
            test_data["model"] = "wrong-model"
            response = requests.post(embeddings_url, json=test_data)
            if response.status_code == 400:
                print("✅ Invalid model properly rejected")
                print("Error message:", response.json().get("error"))
            else:
                print("❌ Invalid model not properly rejected")
        else:
            print("❌ Could not get available models")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_v1_embeddings_endpoint()
    test_model_validation()