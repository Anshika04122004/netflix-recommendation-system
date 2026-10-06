"""
Test script to verify all FastAPI endpoints
"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_all():
    print("Testing GET / ...")
    r0 = client.get("/")
    assert r0.status_code == 200
    print(f"  [OK] Status: {r0.status_code}, Length: {len(r0.text)}")

    print("Testing GET /api/overview ...")
    r1 = client.get("/api/overview")
    assert r1.status_code == 200
    print(f"  [OK] Status: {r1.status_code}, Total titles: {r1.json().get('total_titles')}")

    print("Testing GET /api/clusters ...")
    r2 = client.get("/api/clusters")
    assert r2.status_code == 200
    print(f"  [OK] Status: {r2.status_code}, Clusters: {len(r2.json())}")

    print("Testing GET /api/catalog?page=1&page_size=5 ...")
    r3 = client.get("/api/catalog?page=1&page_size=5")
    assert r3.status_code == 200
    items = r3.json().get("items", [])
    print(f"  [OK] Status: {r3.status_code}, Items returned: {len(items)}")

    print("Testing POST /api/predict ...")
    payload = {
        "title": "Quantum Heist",
        "type": "Movie",
        "release_year": 2023,
        "duration": "115 min",
        "rating": "TV-MA",
        "country": "United States",
        "listed_in": "Action & Adventure, Comedies"
    }
    r4 = client.post("/api/predict", json=payload)
    assert r4.status_code == 200
    res = r4.json()
    print(f"  [OK] Status: {r4.status_code}, Predicted Cluster: {res.get('predicted_cluster_id')}, Nearest: {len(res.get('nearest_neighbors', []))}")

    print("Testing GET /api/strategic-insights ...")
    r5 = client.get("/api/strategic-insights")
    assert r5.status_code == 200
    print(f"  [OK] Status: {r5.status_code}, Gaps: {len(r5.json().get('content_gaps', []))}")

    print("\n>>> ALL API ENDPOINTS PASSED WITH 100% SUCCESS! <<<")

if __name__ == "__main__":
    test_all()
