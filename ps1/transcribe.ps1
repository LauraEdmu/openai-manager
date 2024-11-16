# Define approved PowerShell verb-named functions

function Get-OpenAIKey {
    param (
        [string]$KeyPath = "openai.priv"
    )

    if (-Not (Test-Path -Path $KeyPath)) {
        Write-Error "API key file not found at path: $KeyPath"
        return $null
    }

    try {
        $apiKey = Get-Content -Path $KeyPath -ErrorAction Stop
        Write-Host "API key successfully loaded" -ForegroundColor Green
        return $apiKey
    } catch {
        Write-Error "Failed to read API key file: $_"
        return $null
    }
}

function Invoke-AudioTranscription {
    param (
        [string]$FilePath,
        [string]$ApiKey,
        [string]$Model = "whisper-1"
    )

    if (-Not (Test-Path -Path $FilePath)) {
        Write-Error "Audio file not found at path: $FilePath"
        return
    }

    try {
        # Prepare the REST API endpoint and headers
        $uri = "https://api.openai.com/v1/audio/transcriptions"
        $headers = @{
            "Authorization" = "Bearer $ApiKey"
        }

        # Read the file as a binary stream
        $fileStream = [System.IO.File]::OpenRead($FilePath)
        $fileBytes = [byte[]]::new($fileStream.Length)
        $fileStream.Read($fileBytes, 0, $fileBytes.Length)
        $fileStream.Close()

        # Create a boundary for the multipart form-data
        $boundary = "----WebKitFormBoundary" + [System.Guid]::NewGuid().ToString("N")

        # Build the multipart form-data body
        $bodyLines = @(
            "--$boundary"
            "Content-Disposition: form-data; name=`"model`""
            ""
            $Model
            "--$boundary"
            "Content-Disposition: form-data; name=`"file`"; filename=`"$(Split-Path -Leaf $FilePath)`""
            "Content-Type: audio/mpeg"
            ""
        )
        $bodyStart = [System.Text.Encoding]::UTF8.GetBytes(($bodyLines -join "`r`n") + "`r`n")
        $bodyEnd = [System.Text.Encoding]::UTF8.GetBytes("`r`n--$boundary--`r`n")

        # Concatenate body start, file bytes, and body end
        $body = New-Object System.IO.MemoryStream
        $body.Write($bodyStart, 0, $bodyStart.Length)
        $body.Write($fileBytes, 0, $fileBytes.Length)
        $body.Write($bodyEnd, 0, $bodyEnd.Length)
        $body.Seek(0, [System.IO.SeekOrigin]::Begin)

        # Set the content type header for multipart form-data
        $headers["Content-Type"] = "multipart/form-data; boundary=$boundary"

        # Send the POST request
        $response = Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -Body $body.ToArray()

        if ($response -and $response.text) {
            Write-Host "Transcription completed successfully!" -ForegroundColor Green
            return $response.text
        } else {
            Write-Error "Failed to retrieve transcription text."
            return
        }
    } catch {
        Write-Error "Error during transcription: $_"
    }
}

function Save-TranscriptionToFile {
    param (
        [string]$Text,
        [string]$FilePath
    )

    try {
        $outputFilePath = [System.IO.Path]::ChangeExtension($FilePath, ".txt")
        Set-Content -Path $outputFilePath -Value $Text -Encoding UTF8
        Write-Host "Transcription saved to: $outputFilePath" -ForegroundColor Green
    } catch {
        Write-Error "Failed to save transcription to file: $_"
    }
}

# Main execution
Write-Host "Enter the path to your audio file (e.g., C:\path\to\file.mp3):"
$audioFilePath = Read-Host -Prompt "Audio File Path"

# Resolve the file path to an absolute path
$resolvedPath = Resolve-Path -Path $audioFilePath -ErrorAction SilentlyContinue
if (-Not $resolvedPath) {
    Write-Error "Invalid file path provided. Exiting program."
    exit 1
}

# Define the API key path
$apiKeyPath = "openai.priv"

# Load the API key
$apiKey = Get-OpenAIKey -KeyPath $apiKeyPath
if (-Not $apiKey) {
    Write-Error "Cannot proceed without API key."
    exit 1
}

# Transcribe the audio file
$transcription = Invoke-AudioTranscription -FilePath $resolvedPath.Path -ApiKey $apiKey
if ($transcription) {
    Write-Output "Transcription Result:"
    Write-Output $transcription

    # Save the transcription to a file
    Save-TranscriptionToFile -Text $transcription -FilePath $resolvedPath.Path
}
