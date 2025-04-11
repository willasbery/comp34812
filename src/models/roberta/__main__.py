import logging

# Configure logging right at the start
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
                    # Optional: Add filename='path/to/your/log/file.log' if you want to log to a file

# Now import and run the main function
from src.models.roberta.main import main

if __name__ == "__main__":
    main()