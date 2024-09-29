# Problem Recommender System using KGAT in Codeforces

## How to create the dataset :books:

1. Move to the `src/dataset` directory

2. Create a `.env` file with the following content (refer to `.env.example`)

    - `CODEFORCES_API_KEY`: Codeforces API key
    - `CODEFORCES_API_SECRET`: Codeforces API secret

3. Run the following command

    - Small dataset

        ```bash
        > make create-sm
        ```
    
    - Full dataset

        ```bash
        > make create
        ```

## How to run the system :rocket:

1. Install the required libraries
   
    ```bash
    > make install
    ```

2. unzip the large dataset

    ```bash
    > unzip dataset/users-submission-history.json.zip -d dataset/
    ```

3. Run the system
   
    ```bash
    > make train ${OPTION}
    ```

    ```bash
    > make predict ${OPTION}
    ```

    - Option
      - `--sm`: Use the small dataset