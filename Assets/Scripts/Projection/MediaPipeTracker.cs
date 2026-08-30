using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using UnityEngine;

using Apt.Unity.Projection;

public class MediaPipeTracker : TrackerBase
{
    public Matrix4x4 FaceMatrix { get; private set; }
        = Matrix4x4.identity;

    public Vector3[] FaceLandmarks { get; private set; }
        = new Vector3[LANDMARK_COUNT];

    [SerializeField] private Camera debugCamera;
    [SerializeField] private bool debugDraw = false;
    [SerializeField] private float faceDistance = 1.0f;
    [SerializeField] private float depthScale = 1.0f;
    [SerializeField] private float debugPointSize = 0.003f;

    private const uint MAGIC = 0x46414345;
    private const uint VERSION = 2;

    private const int HEADER_SIZE = 16;

    private const int MATRIX_FLOAT_COUNT = 16;
    private const int MATRIX_SIZE = MATRIX_FLOAT_COUNT * 4;

    private const int LANDMARK_COUNT = 478;
    private const int LANDMARK_COMPONENTS = 3;
    private const int LANDMARK_FLOAT_COUNT =
        LANDMARK_COUNT * LANDMARK_COMPONENTS;
    private const int LANDMARK_SIZE =
        LANDMARK_FLOAT_COUNT * 4;

    private const int SLOT_SEQ_SIZE = 4;

    private const int MATRIX_OFFSET_IN_SLOT =
        SLOT_SEQ_SIZE;

    private const int LANDMARK_OFFSET_IN_SLOT =
        MATRIX_OFFSET_IN_SLOT + MATRIX_SIZE;

    private const int SLOT_SIZE =
        SLOT_SEQ_SIZE +
        MATRIX_SIZE +
        LANDMARK_SIZE;

    private const int TOTAL_SIZE =
        HEADER_SIZE + SLOT_SIZE * 2;

    private const int ACTIVE_INDEX_OFFSET = 8;

    private FileStream fileStream;
    private MemoryMappedFile mappedFile;
    private MemoryMappedViewAccessor accessor;

    private bool connected;

    private readonly float[] matrixValues =
        new float[MATRIX_FLOAT_COUNT];

    private readonly float[] landmarkValues =
        new float[LANDMARK_FLOAT_COUNT];

    private string sharedFile;

    private void Update()
    {
        if (!connected)
        {
            TryConnect();
            return;
        }

        if (!TryReadFaceData())
            return;

        if (debugCamera == null)
            debugCamera = Camera.main;

        if (debugCamera == null)
            return;

        if (debugDraw)
            DrawFaceLandmarks();
        
        IsTracking = true; 
        
        if (IsTracking)
        {
            SecondsHasBeenTracked += Time.deltaTime;
            Vector3 pos = FaceMatrix.GetPosition(); 
            //pos.y += 4.0f;
            translation = pos; 
        }
    }

    private void DrawFaceLandmarks()
    {
        for (int i = 0; i < LANDMARK_COUNT; i++)
        {
            Vector3 landmark = FaceLandmarks[i];

            // MediaPipe:
            // x: 0 = left,   1 = right
            // y: 0 = top,    1 = bottom
            //
            // Unity Viewport:
            // x: 0 = left,   1 = right
            // y: 0 = bottom, 1 = top
            //
            // So Y needs to be flipped.

            float viewportX = landmark.x;
            float viewportY = 1.0f - landmark.y;

            // MediaPipe Z is relative depth, not Unity world depth.
            // Negative Z generally means toward the camera.
            float distance =
                faceDistance +
                landmark.z * depthScale;

            Vector3 worldPoint =
                debugCamera.ViewportToWorldPoint(
                    new Vector3(
                        viewportX,
                        viewportY,
                        distance
                    )
                );

            DrawDebugPoint(
                worldPoint,
                debugPointSize,
                Color.red
            );
        }
    }

    private void DrawDebugPoint(
        Vector3 point,
        float size,
        Color color
    )
    {
        Debug.DrawLine(
            point - Vector3.right * size,
            point + Vector3.right * size,
            color
        );

        Debug.DrawLine(
            point - Vector3.up * size,
            point + Vector3.up * size,
            color
        );

        Debug.DrawLine(
            point - Vector3.forward * size,
            point + Vector3.forward * size,
            color
        );
    }

    private void Awake()
    {
        sharedFile = Path.Combine(
            Path.GetTempPath(),
            "mediapipe_face_data.bin"
        );
    }

    private void Start()
    {
        TryConnect();
    }

    private long SlotOffset(int slot)
    {
        return HEADER_SIZE + slot * SLOT_SIZE;
    }

    private void TryConnect()
    {
        if (!File.Exists(sharedFile))
            return;

        try
        {
            fileStream = new FileStream(
                sharedFile,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite
            );

            mappedFile = MemoryMappedFile.CreateFromFile(
                fileStream,
                null,
                TOTAL_SIZE,
                MemoryMappedFileAccess.Read,
                HandleInheritability.None,
                false
            );

            accessor = mappedFile.CreateViewAccessor(
                0,
                TOTAL_SIZE,
                MemoryMappedFileAccess.Read
            );

            uint magic = accessor.ReadUInt32(0);
            uint version = accessor.ReadUInt32(4);

            if (magic != MAGIC)
                throw new Exception("Invalid shared memory magic.");

            if (version != VERSION)
                throw new Exception(
                    $"Unsupported version: {version}"
                );

            connected = true;
        }
        catch (Exception e)
        {
            Debug.LogError(e);
            Cleanup();
        }
    }

    private bool TryReadFaceData()
    {
        if (!connected)
            return false;

        for (int attempt = 0; attempt < 3; attempt++)
        {
            uint active =
                accessor.ReadUInt32(ACTIVE_INDEX_OFFSET);

            if (active > 1)
                continue;

            long slotOffset =
                SlotOffset((int)active);

            uint sequenceBefore =
                accessor.ReadUInt32(slotOffset);

            if ((sequenceBefore & 1) != 0)
                continue;

            long matrixOffset =
                slotOffset + MATRIX_OFFSET_IN_SLOT;

            accessor.ReadArray(
                matrixOffset,
                matrixValues,
                0,
                MATRIX_FLOAT_COUNT
            );

            long landmarkOffset =
                slotOffset + LANDMARK_OFFSET_IN_SLOT;

            accessor.ReadArray(
                landmarkOffset,
                landmarkValues,
                0,
                LANDMARK_FLOAT_COUNT
            );

            uint sequenceAfter =
                accessor.ReadUInt32(slotOffset);

            if (
                sequenceBefore != sequenceAfter ||
                (sequenceAfter & 1) != 0
            )
            {
                continue;
            }

            Matrix4x4 m = new Matrix4x4();

            m.m00 = matrixValues[0];
            m.m01 = matrixValues[1];
            m.m02 = matrixValues[2];
            m.m03 = matrixValues[3];

            m.m10 = matrixValues[4];
            m.m11 = matrixValues[5];
            m.m12 = matrixValues[6];
            m.m13 = matrixValues[7];

            m.m20 = matrixValues[8];
            m.m21 = matrixValues[9];
            m.m22 = matrixValues[10];
            m.m23 = matrixValues[11];

            m.m30 = matrixValues[12];
            m.m31 = matrixValues[13];
            m.m32 = matrixValues[14];
            m.m33 = matrixValues[15];

            for (int i = 0; i < LANDMARK_COUNT; i++)
            {
                int offset = i * 3;

                FaceLandmarks[i] = new Vector3(
                    landmarkValues[offset],
                    landmarkValues[offset + 1],
                    landmarkValues[offset + 2]
                );
            }

            FaceMatrix = m;

            return true;
        }

        return false;
    }

    private void Cleanup()
    {
        connected = false;

        accessor?.Dispose();
        accessor = null;

        mappedFile?.Dispose();
        mappedFile = null;

        fileStream?.Dispose();
        fileStream = null;
    }

    private void OnDestroy()
    {
        Cleanup();
    }
}