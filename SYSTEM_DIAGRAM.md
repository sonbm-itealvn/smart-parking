# Sơ Đồ Hệ Thống và Luồng Dữ Liệu - Smart Parking System

## 1. Kiến Trúc Hệ Thống Tổng Quan

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Application]
        MOBILE[Mobile App]
        CAMERA[Camera System]
    end

    subgraph "API Gateway"
        FASTAPI[FastAPI Server<br/>Port 8000]
        CORS[CORS Middleware]
    end

    subgraph "Router Layer"
        PARKING_R[Parking Router<br/>/parking-space]
        LICENSE_R[License Router<br/>/license-plate]
        HISTORY_R[History Router<br/>/api/v1/history]
        TICKET_R[Ticket Router<br/>/api/v1/tickets]
        SPOT_R[Spot Router<br/>/api/v1/spots]
        DASHBOARD_R[Dashboard Router<br/>/api/v1/dashboard]
    end

    subgraph "Service Layer"
        PARKING_S[Parking Service<br/>YOLO Detection]
        LICENSE_S[License Plate Service<br/>YOLO + TrOCR]
        HISTORY_S[History Service]
        TICKET_S[Ticket Service]
        SPOT_S[Spot Service]
        PRESENCE_S[Presence Service]
        TRACKER[Centroid Tracker]
    end

    subgraph "AI Models"
        YOLO_PARKING[YOLO Parking Model<br/>models/parking/best.pt]
        YOLO_LICENSE[YOLO License Model<br/>models/license_plate/best.pt]
        TROCR[TrOCR Model<br/>microsoft/trocr-base-printed]
    end

    subgraph "Database"
        MONGO[(MongoDB<br/>Collections:<br/>- vehicle_events<br/>- tickets<br/>- parking_spots<br/>- active_vehicles)]
    end

    subgraph "Storage"
        LOGS[License Logs<br/>CSV File]
    end

    WEB --> FASTAPI
    MOBILE --> FASTAPI
    CAMERA --> FASTAPI
    FASTAPI --> CORS
    CORS --> PARKING_R
    CORS --> LICENSE_R
    CORS --> HISTORY_R
    CORS --> TICKET_R
    CORS --> SPOT_R
    CORS --> DASHBOARD_R

    PARKING_R --> PARKING_S
    LICENSE_R --> LICENSE_S
    HISTORY_R --> HISTORY_S
    TICKET_R --> TICKET_S
    SPOT_R --> SPOT_S
    LICENSE_S --> PRESENCE_S

    PARKING_S --> YOLO_PARKING
    PARKING_S --> TRACKER
    LICENSE_S --> YOLO_LICENSE
    LICENSE_S --> TROCR

    HISTORY_S --> MONGO
    TICKET_S --> MONGO
    SPOT_S --> MONGO
    PRESENCE_S --> MONGO
    LICENSE_S --> LOGS
```

## 2. Luồng Dữ Liệu - License Plate Detection

```mermaid
sequenceDiagram
    participant Client
    participant API as License Router
    participant Service as License Plate Service
    participant YOLO as YOLO License Model
    participant OCR as TrOCR Model
    participant Presence as Presence Service
    participant History as History Service
    participant DB as MongoDB

    Client->>API: POST /license-plate/detect<br/>(Image/URL)
    API->>Service: detect_license_plates(frame)
    
    Service->>YOLO: Detect license plates<br/>(YOLO Model)
    YOLO-->>Service: Bounding boxes
    
    loop For each detected plate
        Service->>Service: Crop plate region
        Service->>Service: Preprocess image<br/>(5 methods: original, CLAHE, binary, adaptive, morph)
        Service->>Service: Split image (top/bottom)
        
        loop For each preprocessed image
            Service->>OCR: TrOCR OCR
            OCR-->>Service: Text + Confidence
        end
        
        Service->>Service: Validate plate text<br/>(_is_valid_plate_text)
        Service->>Service: Combine top/bottom results
        Service->>Service: Format plate text
    end
    
    Service-->>API: Detection results
    
    API->>Presence: Check if vehicle present
    Presence-->>API: Entry/Exit status
    
    API->>History: Record vehicle event
    History->>DB: Save to vehicle_events
    
    alt Entry Event
        API->>Presence: Mark entry
        Presence->>DB: Update active_vehicles
    else Exit Event
        API->>Presence: Mark exit
        Presence->>DB: Remove from active_vehicles
    end
    
    API-->>Client: JSON/Image response<br/>(Plate text, annotated image)
```

## 3. Luồng Dữ Liệu - Parking Space Recommendation

```mermaid
sequenceDiagram
    participant Client
    participant API as Parking Router
    participant Service as Parking Service
    participant YOLO as YOLO Parking Model
    participant Tracker as Centroid Tracker
    participant DB as MongoDB

    Client->>API: POST /parking-space/recommend<br/>(Image/Video)
    API->>Service: recommend_from_frame(frame)
    
    Service->>Service: Resize frame<br/>(max_width: 960px)
    Service->>YOLO: Detect parking spots<br/>(YOLO Model)
    YOLO-->>Service: Detected spots (occupied/empty)
    
    Service->>Service: Filter empty spots
    Service->>Service: Calculate distances<br/>(from entry point)
    Service->>Service: Find closest empty spot
    
    alt Video Processing
        Service->>Tracker: Update tracker<br/>(Centroid tracking)
        Tracker-->>Service: Stable positions
    end
    
    Service->>Service: Draw path to spot<br/>(Using landmarks)
    Service->>Service: Annotate frame<br/>(Bounding boxes, path)
    
    Service-->>API: Annotated image/PNG
    API-->>Client: Response (JSON/Image)
```

## 4. Luồng Dữ Liệu - Ticket Management

```mermaid
sequenceDiagram
    participant Client
    participant API as Ticket Router
    participant Service as Ticket Service
    participant History as History Service
    participant DB as MongoDB

    Client->>API: GET /api/v1/tickets<br/>(Query: status, plate, etc.)
    API->>Service: list_tickets(filters)
    Service->>DB: Query tickets collection
    DB-->>Service: Ticket list
    Service-->>API: Formatted tickets
    API-->>Client: JSON response

    Note over Client,DB: Create Ticket Flow
    Client->>API: POST /api/v1/tickets<br/>(TicketCreate)
    API->>Service: create_ticket(payload)
    
    Service->>History: Get vehicle entry time
    History->>DB: Query vehicle_events
    DB-->>History: Entry event
    History-->>Service: Entry timestamp
    
    Service->>Service: Calculate duration
    Service->>Service: Calculate fee<br/>(rate_hour, first_hour_rate, max_daily)
    Service->>DB: Create ticket document
    DB-->>Service: Created ticket
    Service-->>API: Ticket data
    API-->>Client: JSON response
```

## 5. Luồng Dữ Liệu - Vehicle Entry/Exit

```mermaid
stateDiagram-v2
    [*] --> VehicleDetected: Camera detects vehicle
    
    VehicleDetected --> CheckPresence: License plate recognized
    
    CheckPresence --> Entry: Vehicle NOT in system
    CheckPresence --> Exit: Vehicle IS in system
    
    Entry --> RecordEntry: Create entry event
    RecordEntry --> UpdatePresence: Mark vehicle present
    UpdatePresence --> CreateTicket: Start parking session
    CreateTicket --> [*]
    
    Exit --> RecordExit: Create exit event
    RecordExit --> CalculateFee: Calculate parking duration
    CalculateFee --> UpdateTicket: Update ticket with fee
    UpdateTicket --> RemovePresence: Remove from active vehicles
    RemovePresence --> [*]
    
    note right of CheckPresence
        Query active_vehicles
        collection in MongoDB
    end note
    
    note right of CreateTicket
        Ticket status: ACTIVE
        Start time recorded
    end note
    
    note right of UpdateTicket
        Ticket status: CLOSED
        Fee calculated
    end note
```

## 6. Cấu Trúc Dữ Liệu MongoDB

```mermaid
erDiagram
    VEHICLE_EVENTS ||--o{ TICKETS : "triggers"
    ACTIVE_VEHICLES ||--o{ VEHICLE_EVENTS : "tracks"
    PARKING_SPOTS ||--o{ VEHICLE_EVENTS : "associates"
    
    VEHICLE_EVENTS {
        string id PK
        string plate
        string event_type
        datetime timestamp
        string spot_id FK
        string source
    }
    
    TICKETS {
        string id PK
        string plate
        datetime entry_time
        datetime exit_time
        float duration_hours
        float fee
        string status
        string spot_id FK
    }
    
    ACTIVE_VEHICLES {
        string plate PK
        string event_id FK
        datetime entry_time
        string spot_id FK
        string source
    }
    
    PARKING_SPOTS {
        string id PK
        string name
        string status
        float x
        float y
        float width
        float height
    }
```

## 7. Chi Tiết License Plate Detection Pipeline

```mermaid
flowchart TD
    START[Input: Image/Frame] --> YOLO[YOLO License Detection<br/>Confidence: 0.6]
    YOLO --> CROP[Crop License Plate Region]
    
    CROP --> PREPROCESS[Preprocess Image]
    PREPROCESS --> P1[Method 1: Original RGB]
    PREPROCESS --> P2[Method 2: CLAHE Enhanced]
    PREPROCESS --> P3[Method 3: Binary Threshold]
    PREPROCESS --> P4[Method 4: Adaptive Threshold]
    PREPROCESS --> P5[Method 5: Morphological]
    
    PREPROCESS --> SPLIT[Split Image]
    SPLIT --> TOP[Top Half<br/>First Line]
    SPLIT --> BOTTOM[Bottom Half<br/>Second Line]
    
    P1 --> OCR1[TrOCR OCR]
    P2 --> OCR2[TrOCR OCR]
    P3 --> OCR3[TrOCR OCR]
    P4 --> OCR4[TrOCR OCR]
    P5 --> OCR5[TrOCR OCR]
    TOP --> OCR6[TrOCR OCR]
    BOTTOM --> OCR7[TrOCR OCR]
    
    OCR1 --> VALIDATE1[Validate Text]
    OCR2 --> VALIDATE2[Validate Text]
    OCR3 --> VALIDATE3[Validate Text]
    OCR4 --> VALIDATE4[Validate Text]
    OCR5 --> VALIDATE5[Validate Text]
    OCR6 --> VALIDATE6[Validate Text]
    OCR7 --> VALIDATE7[Validate Text]
    
    VALIDATE1 --> COMBINE[Combine Results]
    VALIDATE2 --> COMBINE
    VALIDATE3 --> COMBINE
    VALIDATE4 --> COMBINE
    VALIDATE5 --> COMBINE
    VALIDATE6 --> COMBINE
    VALIDATE7 --> COMBINE
    
    COMBINE --> MERGE[Merge Top + Bottom]
    MERGE --> FORMAT[Format Plate Text<br/>XX-YZ NNNN]
    FORMAT --> BEST[Select Best Result<br/>Highest Confidence]
    BEST --> OUTPUT[Output: Plate Text + Confidence]
    
    style YOLO fill:#e1f5ff
    style OCR1 fill:#fff4e1
    style OCR2 fill:#fff4e1
    style OCR3 fill:#fff4e1
    style OCR4 fill:#fff4e1
    style OCR5 fill:#fff4e1
    style OCR6 fill:#fff4e1
    style OCR7 fill:#fff4e1
    style VALIDATE1 fill:#ffe1f5
    style VALIDATE2 fill:#ffe1f5
    style VALIDATE3 fill:#ffe1f5
    style VALIDATE4 fill:#ffe1f5
    style VALIDATE5 fill:#ffe1f5
    style VALIDATE6 fill:#ffe1f5
    style VALIDATE7 fill:#ffe1f5
    style OUTPUT fill:#e1ffe1
```

## 8. API Endpoints Overview

```mermaid
graph LR
    subgraph "Parking Space APIs"
        P1[POST /parking-space/recommend]
        P2[POST /parking-space/recommend-video]
        P3[POST /parking-space/annotate-video]
        P4[POST /parking-space/detect-vehicles]
    end

    subgraph "License Plate APIs"
        L1[POST /license-plate/detect]
        L2[GET /license-plate/logs]
    end

    subgraph "Management APIs"
        H1[GET /api/v1/history]
        H2[POST /api/v1/history]
        T1[GET /api/v1/tickets]
        T2[POST /api/v1/tickets]
        S1[GET /api/v1/spots]
        S2[POST /api/v1/spots]
        D1[GET /api/v1/dashboard]
    end

    style P1 fill:#e1f5ff
    style P2 fill:#e1f5ff
    style P3 fill:#e1f5ff
    style P4 fill:#e1f5ff
    style L1 fill:#fff4e1
    style L2 fill:#fff4e1
    style H1 fill:#ffe1f5
    style T1 fill:#e1ffe1
    style S1 fill:#f5e1ff
```

## 9. Technology Stack

```
┌─────────────────────────────────────────┐
│         Client Applications              │
│  (Web, Mobile, Camera Systems)          │
└─────────────────┬───────────────────────┘
                  │ HTTP/REST
┌─────────────────▼───────────────────────┐
│         FastAPI Backend                  │
│  - Python 3.13                           │
│  - FastAPI Framework                     │
│  - Uvicorn ASGI Server                   │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐   ┌────▼────┐   ┌────▼────┐
│ YOLO  │   │  TrOCR  │   │ MongoDB │
│ Models│   │  Model  │   │ Database│
└───────┘   └─────────┘   └─────────┘
    │             │             │
    │             │             │
┌───▼─────────────▼─────────────▼───┐
│      AI/ML Models                 │
│  - YOLOv8 (Ultralytics)           │
│  - TrOCR (Microsoft)              │
│  - PyTorch                        │
└───────────────────────────────────┘
```

## 10. Data Flow Summary

### License Plate Detection Flow:
1. **Input**: Image/Video frame → FastAPI Router
2. **Detection**: YOLO detects license plate regions
3. **OCR**: TrOCR reads text from each detected region
4. **Validation**: Filter invalid results
5. **Storage**: Save to MongoDB (vehicle_events, active_vehicles)
6. **Output**: Annotated image + plate text

### Parking Recommendation Flow:
1. **Input**: Image/Video → FastAPI Router
2. **Detection**: YOLO detects parking spots (occupied/empty)
3. **Analysis**: Calculate distances, find closest empty spot
4. **Tracking**: Centroid tracker for video processing
5. **Visualization**: Draw path and annotate frame
6. **Output**: Annotated image with recommendation

### Ticket Management Flow:
1. **Entry**: Vehicle detected → Create entry event → Start ticket
2. **Tracking**: Monitor active vehicles in MongoDB
3. **Exit**: Vehicle exits → Calculate duration → Calculate fee
4. **Storage**: Save ticket with fee to MongoDB
5. **Query**: Retrieve tickets by status, plate, date range

---

**Ghi chú**: 
- Tất cả các sơ đồ được tạo bằng Mermaid syntax
- Có thể render trên GitHub, GitLab, hoặc các Markdown viewer hỗ trợ Mermaid
- Để xem trực quan, sử dụng: https://mermaid.live/

