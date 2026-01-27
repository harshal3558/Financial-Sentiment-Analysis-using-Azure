from src.FSAAZ.exception import CustomException
from src.FSAAZ.logger import logging
from src.FSAAZ.components.data_ingestion import DataIngestion
from src.FSAAZ.components.data_ingestion import DataIngestionConfig
import sys

if __name__ == "__main__":
    logging.info("The execution has started.")

    try:
        # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info('Custom Exception')
        raise CustomException(e,sys)