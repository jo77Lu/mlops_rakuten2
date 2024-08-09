from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI()

DATABASE_URL = "sqlite:///./test.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String)

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True, index=True)
    images = Column(String)
    description = Column(String)
    category = Column(String)
    confidence = Column(String)

Base.metadata.create_all(bind=engine)

class ProductCreate(BaseModel):
    images: str
    description: str

class ProductUpdate(BaseModel):
    images: str = None
    description: str = None

@app.post('/products', status_code=status.HTTP_201_CREATED)
def create_product(product: ProductCreate):
    db = SessionLocal()
    db_product = Product(images=product.images, description=product.description)
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return {"product_id": db_product.id, "status": "Product created"}

@app.get('/products/{product_id}')
def get_product(product_id: int):
    db = SessionLocal()
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    return db_product

@app.put('/products/{product_id}')
def update_product(product_id: int, product: ProductUpdate):
    db = SessionLocal()
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    if product.images:
        db_product.images = product.images
    if product.description:
        db_product.description = product.description
    
    db.commit()
    return {"status": "Product updated"}

@app.post('/classify')
def classify_product(product_id: int):
    db = SessionLocal()
    db_product = db.query(Product).filter(Product.id == product_id).first()
    if not db_product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    # Dummy classification logic
    db_product.category = "dummy_category"
    db_product.confidence = "0.9"
    db.commit()
    return {"product_id": db_product.id, "category": db_product.category, "confidence": db_product.confidence}

@app.get('/logs')
def get_logs(start_date: str, end_date: str):
    # Dummy log retrieval
    logs = [
        {"date": "2023-01-01", "log": "Log entry 1"},
        {"date": "2023-01-02", "log": "Log entry 2"},
    ]
    return logs

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)