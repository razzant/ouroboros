        payload = {
            "client_id": "test_id",
            "client_secret": "super_secret_val",
            "tokens": {
                "access_token": "secret_access_tok",
                "refresh_token": "secret_refresh_tok",
            },
            "meeting_id": "01HW123",
            {
                "client_id": "dcr_client_12345",
                "client_secret": "dcr_secret_67890",
                "client_id_issued_at": 1700000000,
            },
        with self.client._file_lock():
            self.client._write_state_locked(
                {"client_credentials": {"client_id": "manual_cid", "client_secret": "manual_sec"}}
            )

    def test_authorization_url_generation(self, mock_http):
        """Verify consent URL contains correct PKCE parameters and stores pending_auth state."""
        mock_http.return_value = (201, {}, {"client_id": "cid_test", "client_secret": "csec_test"})

        url = self.client.get_authorization_url(redirect_uri="urn:ietf:wg:oauth:2.0:oob")
        with self.client._file_lock():
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "pending_auth": {
                    "code_verifier": "test_verifier_string",
            {},
            {
                "access_token": "acc_tok_999",
                "refresh_token": "ref_tok_888",
                "token_type": "Bearer",
                "expires_in": 600,
        with self.client._file_lock():
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "tokens": {
                    "access_token": "expired_access_token",
                    "refresh_token": "initial_refresh_token",
                    "expires_at": int(time.time()) - 10,  # Expired
                },
            {},
            {
                "access_token": "new_access_token_111",
                "refresh_token": "new_rotated_refresh_token_222",
                "expires_in": 600,
            },
        with self.client._file_lock():
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "tokens": {
                    "access_token": "expired_tok",
                    "refresh_token": "revoked_refresh_tok",
                    "expires_at": int(time.time()) - 10,
                },
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test"},
                "tokens": {"access_token": "valid_tok", "expires_at": int(time.time()) + 500},
            })

            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test"},
                "tokens": {"access_token": "valid_tok", "expires_at": int(time.time()) + 500},
            })

            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test"},
                "tokens": {"access_token": "valid_tok", "expires_at": int(time.time()) + 500},
            })

            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test"},
                "tokens": {"access_token": "valid_tok", "expires_at": int(time.time()) + 500},
            })

        with self.client._file_lock():
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "pending_auth": {
                    "code_verifier": "test_verifier_string",
            {},
            {
                "access_token": "acc_tok_url_test",
                "refresh_token": "ref_tok_url_test",
                "token_type": "Bearer",
                "expires_in": 600,
        with self.client._file_lock():
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "tokens": {
                    "access_token": "stale_token",
                    "refresh_token": "valid_refresh",
                    "expires_at": int(time.time()) + 300,
                },
        mock_http.side_effect = [
            (401, {}, {"error": "invalid_token", "message": "Access token expired or revoked"}),
            (200, {}, {"access_token": "fresh_acc_tok", "refresh_token": "fresh_ref_tok", "expires_in": 600}),
            (200, {}, {"meetings": [{"id": "m1", "title": "Post-Refresh Meeting"}]}),
        ]
        with self.client._file_lock():
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "pending_auth": {
                    "code_verifier": "test_verifier_string",
            {},
            {
                "access_token": "acc_tok_state_test",
                "refresh_token": "ref_tok_state_test",
                "token_type": "Bearer",
                "expires_in": 600,
        with self.client._file_lock():
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "pending_auth": {
                    "code_verifier": "test_verifier_string",
        with self.client._file_lock():
            self.client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "tokens": {
                    "access_token": "old_acc",
                    "refresh_token": "spent_ref",
                    "expires_at": int(time.time()) - 10,
                },
            {},
            {
                "access_token": "new_acc_only",
                "expires_in": 600,
            },
                "client_credentials": {"client_id": "cid_test"},
                "tokens": {
                    "access_token": "valid_tok",
                    "refresh_token": "valid_ref",
                    "expires_at": int(time.time()) + 400,
                },
            {},
            {
                "access_token": "plug_acc_tok",
                "refresh_token": "plug_ref_tok",
                "token_type": "Bearer",
                "expires_in": 600,
        with plugin._client._file_lock():
            plugin._client._write_state_locked({
                "client_credentials": {"client_id": "cid_test", "client_secret": "csec_test"},
                "pending_auth": {
                    "code_verifier": "plug_verifier",
        """Verify handle_list_meetings preflight checks readiness and returns consent guidance when unauthenticated."""
        with patch.object(ReadAiClient, "_http_request") as mock_http:
            mock_http.return_value = (200, {}, {"client_id": "dcr_cid", "client_secret": "dcr_sec"})
            list_fn = self.registered_tools["read_ai_list_meetings"]["fn"]
            raw_res = list_fn()
        """Verify handle_get_meeting preflight checks readiness and returns consent guidance when unauthenticated."""
        with patch.object(ReadAiClient, "_http_request") as mock_http:
            mock_http.return_value = (200, {}, {"client_id": "dcr_cid", "client_secret": "dcr_sec"})
            get_fn = self.registered_tools["read_ai_get_meeting"]["fn"]
            raw_res = get_fn(id="01HWXYZ123")
