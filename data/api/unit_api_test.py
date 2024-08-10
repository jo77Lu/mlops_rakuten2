import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .api import app, SessionLocal, Base, engine

"""
api_test.py
--------------------
Ce fichier contient les tests unitaires dedies a valider les cas nominaux et les cas d'erreur
"""

client = TestClient(app)

SQLALCHEMY_DATABASE_URL = "sqlite://"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture
def valid_user():
    return {"username":"", "password":""}

@pytest.fixture
def valid_product():
    return {}

@pytest.fixture
def token():
    response = client.post(
            "/auth/login",
            headers={},
            json=valid_user
        )
    return response.json()["token"]


class TestAuthentication:
    def test_login_valid(app):
        response = client.post(
            "/auth/login",
            headers={},
            json=valid_user
        )
        assert response.status_code == 200
        assert "token" in response.json()
        assert response.json()["token"] is not None

    def test_login_invalid_username(app):
        response = client.post(
            "/auth/login",
            headers={},
            json={"username":"dummy_client", "password":"dummy_password"}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid username or password"

    def test_login_invalid_password(app):
        response = client.post(
            "/auth/login",
            headers={},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid username or password"

    def test_logout_valid_token(app):
        response = client.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json() == {"message": "Successfully logged out"}

    def test_logout_invalid_token(app):
        response = client.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer dummyToken"},
        )
        assert response.status_code == 401
        assert response.json() == {}

class TestProductManagement:
    def test_add_product(app):
        response = client.post(
            "/products",
            headers={"Authorization": f"Bearer {token}"},
            json={}
        )
        assert response.status.code == 201
        assert 'product_id' in response.json() and 'status' in response.json()
        
    def test_add_product_missing_images(app):
        response = client.post(
            "/products",
            headers={"Authorization": f"Bearer {token}"},
            json={}
        )
        assert response.status.code == 401
        assert response.json()["detail"] == "Images are mandatory"
        
    def test_add_product_incorrect_image_path(app):
        response = client.post(
            "/products",
            headers={"Authorization": f"Bearer {token}"},
            json={}
        )
        assert response.status.code == 401
        assert response.json()["detail"] == "Incorrect image path."
        
    def test_add_product_missing_description(app):
        response = client.post(
            "/products",
            headers={"Authorization": f"Bearer {token}"},
            json={}
        )
        assert response.status.code == 401
        assert response.json()["detail"] == "Description is mandatory."
        
    def test_add_product_long_description(app):
        response = client.post(
            "/products",
            headers={"Authorization": f"Bearer {token}"},
            json={}
        )
        assert response.status.code == 401
        assert response.json()["detail"] == "Number of characters exceeded in description."
        
    def test_get_product(app):
        response = client.get(
            f"/products/",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        
class 