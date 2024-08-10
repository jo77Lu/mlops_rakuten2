import time
import pytest
from httpx import Client
from fastapi import status
from app import app, User, Product, SessionLocal

max_description_length = 3000

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
    assert response.json() == {"detail": "Product not found"}

def test_create_product_empty_description(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    response = client.post('/products', 
                           json={"images": "image_data", "description": ""},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json() == {"detail": "Description cannot be empty to create or update a product."}

def test_create_product_description_too_long(client, create_user):
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    long_description = "x" * (max_description_length + 1)  # Max length of description + 1 characters long

    response = client.post(
        '/products', 
        json={"images": "image_data", "description": long_description},
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json() == {"detail": f"Description should have less than {max_description_length} characters."}

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
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json() == {"detail": "At least an image or a description is required to update a product."}

def test_update_product_empty_description(client, create_user, db):
    user = create_user("testuser", "testpass")
    product = Product(images="image_data", description="New product")
    db.add(product)
    db.commit()

    login_response = client.post('/auth/login', json={"username": user.username, "password": user.password})
    token = login_response.json()["token"]

    response = client.put(f'/products/{product.id}', 
                          json={"description": ""},
                          headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json() == {"detail": "Description cannot be empty to create or update a product."}
    
def test_create_product_description_too_long(client, create_user):
    user = create_user("testuser", "testpass")
    product = Product(images="image_data", description="New product")
    db.add(product)
    db.commit()
    
    login_response = client.post('/auth/login', json={"username": user.username, "password": user.password})
    token = login_response.json()["token"]
    
    long_description = "x" * (max_description_length + 1)  # Max length of description + 1 characters long
    
    response = client.put(
        f'/products/{product.id}', 
        json={"description": long_description},
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert response.json() == {"detail": f"Description should have less than {max_description_length} characters."}

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
    assert elapsed_time < 0.5

def test_create_user_and_login(client, create_user):
    username = "testuser"
    password = "testpass"

    # Measure the time to create a user
    start_time = time.time()
    response = client.post('/auth/login', json={"username": username, "password": password})
    create_duration = time.time() - start_time

    assert response.status_code == status.HTTP_200_OK
    assert "token" in response.json()

    token = response.json()["token"]

    print(f"Time taken to create and login a user: {create_duration:.2f} seconds")

def test_create_multiple_products(client, create_user):
    # Create a test user and get the access token
    create_user("testuser", "testpass")
    login_response = client.post('/auth/login', json={"username": "testuser", "password": "testpass"})
    token = login_response.json()["token"]

    # Number of products to create
    num_products = 100

    # Function to create a product
    def create_product(i):
        return client.post(
            '/products', 
            json={"images": f"image_data_{i}", "description": f"Product {i}"},
            headers={"Authorization": f"Bearer {token}"}
        )

    # Measure the time to create products
    start_time = time.time()
    responses = [create_product(i) for i in range(num_products)]
    create_duration = time.time() - start_time

    # Check all responses
    for response in responses:
        assert response.status_code == status.HTTP_201_CREATED
        assert "product_id" in response.json()

    print(f"Time taken to create {num_products} products: {create_duration:.2f} seconds")

def test_create_multiple_users_and_products(client):
    # Number of users and products to create
    num_users = 10
    num_products_per_user = 10

    # Function to create a user and their products
    def create_user_and_products(user_id):
        username = f"user_{user_id}"
        password = f"pass_{user_id}"

        # Create the user
        client.post('/auth/login', json={"username": username, "password": password})
        login_response = client.post('/auth/login', json={"username": username, "password": password})
        token = login_response.json()["token"]

        # Function to create a product
        def create_product(product_id):
            return client.post(
                '/products', 
                json={"images": f"image_data_{product_id}", "description": f"Product {product_id}"},
                headers={"Authorization": f"Bearer {token}"}
            )

        # Create products for the user
        responses = [create_product(i) for i in range(num_products_per_user)]

        # Check all responses
        for response in responses:
            assert response.status_code == status.HTTP_201_CREATED
            assert "product_id" in response.json()

    # Measure the time to create users and products
    start_time = time.time()
    for i in range(num_users):
        create_user_and_products(i)
    total_duration = time.time() - start_time

    print(f"Time taken to create {num_users} users and {num_users * num_products_per_user} products: {total_duration:.2f} seconds")