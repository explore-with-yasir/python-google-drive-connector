# python-google-drive-connector

- Connect with google drive to read files for any given user in organization with username
- Read contents of Google docs, Google spreadsheets, Google presentation & PDFs available in google drive
- Read all files from a given folder
- Do text splitting of these documents
- Perform embedding on the splitted text
- Save embeddings in Chroma vectorDB (individual vectorDB for each customer)
- Query embeddings to find out similarity between given prompt & the stored embeddings, return text with scores (cosine distance scores)
- All services mentioned available as REST APIs
