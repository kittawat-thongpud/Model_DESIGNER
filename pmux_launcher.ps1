Set-Location 'C:\kittawat_ws\Model_DESIGNER'

 = "1"
System.Text.ASCIIEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

python run.py *>&1 | Tee-Object -FilePath 'C:\kittawat_ws\Model_DESIGNER\server.log'
