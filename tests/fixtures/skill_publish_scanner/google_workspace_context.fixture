        url_str = str(request.url)
        if "oauth2.googleapis.com/token" in url_str:
            return httpx.Response(200, json={"access_token": "mock_token", "expires_in": 3600})
        if "sheets.googleapis.com/v4/spreadsheets/sheet123/values/Sheet1!A1" in url_str:
            return httpx.Response(
        url_str = str(request.url)
        if "oauth2.googleapis.com/token" in url_str:
            return httpx.Response(200, json={"access_token": "mock_token", "expires_in": 3600})
        if "values/Sheet1!A1:append" in url_str:
            return httpx.Response(
        url_str = str(request.url)
        if "oauth2.googleapis.com/token" in url_str:
            return httpx.Response(200, json={"access_token": "mock_token", "expires_in": 3600})
        if request.method == "POST" and "docs.googleapis.com/v1/documents" in url_str:
            return httpx.Response(200, json={"documentId": "doc12345", "title": "My New Document"})
        url_str = str(request.url)
        if "oauth2.googleapis.com/token" in url_str:
            return httpx.Response(200, json={"access_token": "mock_token", "expires_in": 3600})
        if "drive/v3/files/template999/copy" in url_str:
            return httpx.Response(200, json={"id": "doc_from_template_777", "name": "Copied Doc"})
        url_str = str(request.url)
        if "oauth2.googleapis.com/token" in url_str:
            return httpx.Response(200, json={"access_token": "mock_token", "expires_in": 3600})
        if "drive/v3/files" in url_str:
            assert "'folder123' in parents" in request.url.params.get("q", "")
        url_str = str(request.url)
        if "oauth2.googleapis.com/token" in url_str:
            return httpx.Response(200, json={"access_token": "mock_token", "expires_in": 3600})
        if url_str.endswith("drive/v3/files/file_doc?fields=id%2C+name%2C+mimeType%2C+size&supportsAllDrives=true"):
            return httpx.Response(
        url_str = str(request.url)
        if "oauth2.googleapis.com/token" in url_str:
            return httpx.Response(200, json={"access_token": "mock_token", "expires_in": 3600})
        if "drive/v3/files/file_long" in url_str and "export" not in url_str:
            return httpx.Response(
        url_str = str(request.url)
        if "oauth2.googleapis.com/token" in url_str:
            return httpx.Response(200, json={"access_token": "mock_token", "expires_in": 3600})
        if "drive/v3/files/file_image" in url_str:
            return httpx.Response(
