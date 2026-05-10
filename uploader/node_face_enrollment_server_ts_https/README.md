# TypeScript HTTPS Face Enrollment Server

## 1. Install dependencies
```powershell
npm install
```

## 2. Find your PC Wi-Fi IP
```powershell
ipconfig
```
Example: `192.168.1.35`

## 3. Install mkcert
```powershell
winget install FiloSottile.mkcert
```
Close and reopen PowerShell.

## 4. Create local certificate authority
```powershell
mkcert -install
```

## 5. Generate certificate
Replace the IP with your real Wi-Fi IP:
```powershell
mkcert -cert-file certs\local-cert.pem -key-file certs\local-key.pem 192.168.1.35 localhost 127.0.0.1
```

## 6. Create `.env`
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Edit `.env` if needed:
```text
PORT=3000
UPLOAD_DIR=C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Face Recognition ex\images
HTTPS_KEY_FILE=certs\local-key.pem
HTTPS_CERT_FILE=certs\local-cert.pem
```

## 7. Run
```powershell
npm run dev
```

Open on PC:
```text
https://localhost:3000
```

Open on phone:
```text
https://YOUR_PC_WIFI_IP:3000
```

## 8. If the phone cannot connect
Allow port 3000:
```powershell
netsh advfirewall firewall add rule name="Face Enrollment HTTPS 3000" dir=in action=allow protocol=TCP localport=3000
```
