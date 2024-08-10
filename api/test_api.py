import pytest
from httpx import Client
from fastapi import status
from app import app, User, Product, SessionLocal

@pytest.fixture(scope="function")
def db():
    db = SessionLocal()
    yield db
    db.close()

@pytest.fixture(scope="function")
def create_user(db):
    def _create_user(username, password, role="user"):
        user = User(username=username, password=password, role=role)
        db.add(user)
        db.commit()
        return user
    return _create_user

@pytest.fixture(scope="function")
def client():
    with Client(app=app, base_url="http://testserver") as client:
        yield client

def test_login_success(client, create_user):
    create_user("testuser", "testpass")
    response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    assert response.status_code == status.HTTP_200_OK
    assert "token" in response.json()

def test_login_failure(client, create_user):
    create_user("testuser", "testpass")
    response = client.post('/auth/login', json={"username": "testuser", "password": "wrongpass"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Invalid username or password"}

def test_login_with_invalid_input(client):
    response = client.post('/auth/login', json={"username": "", "password": ""})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": "Invalid username or password"}

def test_login_with_missing_fields(client):
    response = client.post('/auth/login', json={"username": "testuser"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_logout_success(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.post('/auth/logout', headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Successfully logged out"}

def test_logout_invalid_token(client):
    response = client.post('/auth/logout', headers={"Authorization": f"Bearer dummyToken"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_logout_without_token(client):
    response = client.post('/auth/logout')
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_create_product_success(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.post('/products', 
                           json={"images": "image_data", "description": "New product"},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_201_CREATED
    assert "product_id" in response.json()

def test_create_product_missing_fields(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.post('/products', 
                           json={"images": "image_data"},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_create_product_empty_description(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.post('/products', 
                           json={"images": "image_data", "description": ""},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_201_CREATED
    assert "product_id" in response.json()

def test_get_product_success(client, create_user, db):
    user = create_user("testuser", "testpass")
    product = Product(images="image_data", description="New product")
    db.add(product)
    db.commit()

    login_response = client.post('/auth/login', json={"username": user.username, "password": user.password})
    token = login_response.json()["token"]

    response = client.get(f'/products/{product.id}', headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["description"] == "New product"

def test_get_product_not_found(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.get('/products/9999', headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "Product not found"}

def test_update_product_success(client, create_user, db):
    user = create_user("testuser", "testpass")
    product = Product(images="image_data", description="New product")
    db.add(product)
    db.commit()

    login_response = client.post('/auth/login', json={"username": user.username, "password": user.password})
    token = login_response.json()["token"]

    response = client.put(f'/products/{product.id}', 
                          json={"description": "Updated description"},
                          headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "Product updated"}

def test_update_product_not_found(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.put('/products/9999', 
                          json={"description": "Updated description"},
                          headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "Product not found"}

def test_update_product_no_fields(client, create_user, db):
    user = create_user("testuser", "testpass")
    product = Product(images="image_data", description="New product")
    db.add(product)
    db.commit()

    login_response = client.post('/auth/login', json={"username": user.username, "password": user.password})
    token = login_response.json()["token"]

    response = client.put(f'/products/{product.id}', 
                          json={},
                          headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "Product updated"}

def test_classify_product_success(client, create_user, db):
    user = create_user("testuser", "testpass")
    product = Product(images="image_data", description="New product")
    db.add(product)
    db.commit()

    login_response = client.post('/auth/login', json={"username": user.username, "password": user.password})
    token = login_response.json()["token"]

    response = client.post('/classify', json={"product_id": product.id}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["category"] == "dummy_category"
    assert response.json()["confidence"] == "0.9"

def test_classify_product_not_found(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.post('/classify', json={"product_id": 9999}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json() == {"detail": "Product not found"}

def test_get_logs_success(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.get('/logs?start_date=2023-01-01&end_date=2023-01-02', 
                          headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) > 0

def test_get_logs_without_dates(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.get('/logs', headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_login_performance(client, create_user):
    create_user("testuser", "testpass")

    import time
    start_time = time.time()

    response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    assert response.status_code == status.HTTP_200_OK

    elapsed_time = time.time() - start_time
    assert elapsed_time < 0.5  # Example threshold
