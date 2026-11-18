#!/usr/bin/swift
// Local, unreviewed prose extraction only. This does not parse mahjong tiles.
import Foundation
import Vision
import ImageIO
import CryptoKit
import Darwin

struct OCRFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Arguments {
    let outputDirectory: URL
    let language: String
    let images: [URL]
}

func parseArguments() throws -> Arguments {
    let values = Array(CommandLine.arguments.dropFirst())
    var output: String?
    var language: String?
    var paths: [String] = []
    var index = 0
    var pathsOnly = false
    while index < values.count {
        let value = values[index]
        if value == "--" && !pathsOnly {
            pathsOnly = true
        } else if !pathsOnly && (value == "--outputdir" || value == "--language") {
            guard index + 1 < values.count, !values[index + 1].isEmpty,
                  !values[index + 1].hasPrefix("--") else {
                throw OCRFailure("Missing value for \(value)")
            }
            index += 1
            if value == "--outputdir" {
                guard output == nil else { throw OCRFailure("Duplicate --outputdir") }
                output = values[index]
            } else {
                guard language == nil else { throw OCRFailure("Duplicate --language") }
                language = values[index]
            }
        } else if !pathsOnly && value.hasPrefix("--") {
            throw OCRFailure("Unknown option: \(value)")
        } else {
            paths.append(value)
        }
        index += 1
    }
    guard let output = output, let language = language, !paths.isEmpty else {
        throw OCRFailure("Usage: book_page_ocr.swift --outputdir DIR --language ja-JP|zh-Hans IMAGE.png [IMAGE.png ...]")
    }
    return Arguments(
        outputDirectory: URL(fileURLWithPath: output, isDirectory: true).standardizedFileURL,
        language: language,
        images: paths.map { URL(fileURLWithPath: $0).standardizedFileURL }
    )
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func jsonData(_ object: [String: Any]) throws -> Data {
    var data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    data.append(0x0a)
    return data
}

func outputURL(for input: URL, in directory: URL) -> URL {
    directory.appendingPathComponent(input.deletingPathExtension().lastPathComponent + ".ocr.json")
}

@available(macOS 13.0, *)
func configuredRequest(language: String) -> VNRecognizeTextRequest {
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequestRevision3
    request.recognitionLevel = .accurate
    request.recognitionLanguages = [language]
    request.automaticallyDetectsLanguage = false
    request.usesLanguageCorrection = true
    request.minimumTextHeight = 0.0
    return request
}

@available(macOS 13.0, *)
func run(_ arguments: Arguments) throws {
    let revision = VNRecognizeTextRequestRevision3
    guard VNRecognizeTextRequest.supportedRevisions.contains(Int(revision)) else {
        throw OCRFailure("Vision text recognition revision 3 is unavailable")
    }
    let probe = configuredRequest(language: arguments.language)
    let supported = try probe.supportedRecognitionLanguages()
    guard supported.contains(arguments.language) else {
        throw OCRFailure("Requested language \(arguments.language) is unsupported by accurate revision 3; supported: \(supported.joined(separator: ", "))")
    }

    // Preflight names before recognition. Exclusive writes also protect against races.
    let manager = FileManager.default
    var outputPaths = Set<String>()
    for input in arguments.images {
        var isDirectory: ObjCBool = false
        guard input.pathExtension.lowercased() == "png",
              manager.fileExists(atPath: input.path, isDirectory: &isDirectory),
              !isDirectory.boolValue, manager.isReadableFile(atPath: input.path) else {
            throw OCRFailure("Input must be a readable PNG file: \(input.path)")
        }
        let destination = outputURL(for: input, in: arguments.outputDirectory)
        guard outputPaths.insert(destination.path).inserted else {
            throw OCRFailure("Duplicate output basename: \(destination.lastPathComponent)")
        }
        guard !manager.fileExists(atPath: destination.path) else {
            throw OCRFailure("Refusing to overwrite: \(destination.path)")
        }
    }
    try manager.createDirectory(at: arguments.outputDirectory, withIntermediateDirectories: true)

    let scriptURL = URL(fileURLWithPath: #filePath).standardizedFileURL
    let scriptHash: Any
    if let scriptData = try? Data(contentsOf: scriptURL) {
        scriptHash = sha256(scriptData)
    } else {
        scriptHash = NSNull()
    }
    let visionBundle = Bundle(for: VNRecognizeTextRequest.self)
    let visionVersion: Any = visionBundle.object(forInfoDictionaryKey: "CFBundleVersion") ?? NSNull()
    let runtime: [String: Any] = [
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "vision_bundle_identifier": visionBundle.bundleIdentifier ?? "com.apple.Vision",
        "vision_bundle_version": visionVersion,
        "processor_count": ProcessInfo.processInfo.processorCount,
        "script_path": scriptURL.path,
        "script_sha256": scriptHash,
        "execution": "local Apple Vision; no network calls in this script"
    ]

    for (pageIndex, input) in arguments.images.enumerated() {
        try autoreleasepool {
            let startedAt = ISO8601DateFormatter().string(from: Date())
            let startedClock = DispatchTime.now().uptimeNanoseconds
            // Hash and decode the same bytes; do not trust a file extension alone.
            let inputData = try Data(contentsOf: input)
            guard let source = CGImageSourceCreateWithData(inputData as CFData, nil),
                  let sourceType = CGImageSourceGetType(source),
                  sourceType as String == "public.png", CGImageSourceGetCount(source) == 1,
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw OCRFailure("Cannot decode a single-image PNG: \(input.path)")
            }
            let request = configuredRequest(language: arguments.language)
            // Rendered page PNGs are upright; record this explicit orientation.
            let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
            try handler.perform([request])
            guard let observations = request.results else {
                throw OCRFailure("Vision returned no result collection for: \(input.path)")
            }
            var characterCount = 0
            var blocks: [[String: Any]] = []
            for (observationIndex, observation) in observations.enumerated() {
                let candidate = observation.topCandidates(1).first
                let text: Any
                let confidence: Any
                if let candidate = candidate {
                    text = candidate.string
                    confidence = Double(candidate.confidence)
                    characterCount += candidate.string.count
                } else {
                    text = NSNull()
                    confidence = NSNull()
                }
                let box = observation.boundingBox
                blocks.append([
                    "returned_order_index": observationIndex,
                    "text": text,
                    "candidate_confidence": confidence,
                    "observation_confidence": Double(observation.confidence),
                    "candidate_missing": candidate == nil,
                    "bbox_normalized": [
                        "x": Double(box.origin.x), "y": Double(box.origin.y),
                        "width": Double(box.size.width), "height": Double(box.size.height)
                    ]
                ])
            }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - startedClock) / 1_000_000_000
            let result: [String: Any] = [
                "schema": "riichi-book-prose-ocr-v1",
                "review_status": "unreviewed",
                "training_ready": false,
                "tile_parsing_performed": false,
                "complete_question_assessed": false,
                "scope": "candidate prose blocks; tile-like glyphs may be mistranscribed as text",
                "source": [
                    "image_path": input.path, "image_sha256": sha256(inputData),
                    "image_bytes": inputData.count, "width_pixels": image.width,
                    "height_pixels": image.height, "image_orientation": "up (explicit assumption)",
                    "invocation_input_index": pageIndex
                ],
                "recognition": [
                    "framework": "Apple Vision", "request_revision": Int(revision),
                    "recognition_level": "accurate", "requested_language": arguments.language,
                    "recognition_languages": request.recognitionLanguages,
                    "supported_languages": supported, "automatically_detects_language": false,
                    "uses_language_correction": true, "minimum_text_height": 0.0,
                    "candidate_count_per_observation": 1
                ],
                "runtime": runtime,
                "started_at_utc": startedAt,
                "elapsed_seconds_through_recognition": elapsed,
                "block_count": blocks.count,
                "recognized_character_count": characterCount,
                "block_order": "Vision returned order; reading order not validated",
                "bbox_coordinate_system": "normalized image coordinates; origin lower-left",
                "confidence_interpretation": "Vision recognition score; not measured transcription accuracy",
                "blocks": blocks
            ]
            let destination = outputURL(for: input, in: arguments.outputDirectory)
            let data = try jsonData(result)
            // .withoutOverwriting is exclusive; Foundation forbids combining it with .atomic.
            try data.write(to: destination, options: [.withoutOverwriting])
            let summary: [String: Any] = [
                "output": destination.path, "output_sha256": sha256(data),
                "block_count": blocks.count, "recognized_character_count": characterCount,
                "elapsed_seconds": elapsed, "review_status": "unreviewed"
            ]
            let line = try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys])
            FileHandle.standardOutput.write(line)
            FileHandle.standardOutput.write(Data([0x0a]))
        }
    }
}

do {
    let arguments = try parseArguments()
    if #available(macOS 13.0, *) {
        try run(arguments)
    } else {
        throw OCRFailure("macOS 13 or later is required for pinned Vision revision 3")
    }
} catch {
    FileHandle.standardError.write(Data("OCR failed: \(error)\nCompleted page outputs, if any, were retained; no output was overwritten.\n".utf8))
    exit(1)
}
