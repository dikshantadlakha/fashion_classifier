import os
import base64
import json
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import gradio as gr
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(__file__)
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")
SELECT_SOUND_PATH = os.path.join(SOUNDS_DIR, "universfield-button-124476.mp3")

APP_CSS = r"""
    .intro-banner {
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
        border-radius: 8px;
    }

    .intro-banner::before {
        content: "";
        position: absolute;
        inset: 0;
        background-size: 28px 28px;
        mask-image: linear-gradient(90deg, rgba(0, 0, 0, 0.42), transparent 72%);
        pointer-events: none;
    }

    .intro-banner > * {
        position: relative;
        z-index: 1;
    }

    .intro-banner h1 {
        margin: 0 0 14px;
        width: fit-content;
    }

    .intro-banner p {
        max-width: 760px;
        margin: 0 0 8px;
        line-height: 1.55;
    }

    .intro-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .model-arena {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }

    .model-arena span {
        padding: 9px 12px;
        border-radius: 8px;
        font-size: 0.92rem;
        font-weight: 650;
        transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, color 0.16s ease;
    }

    .model-arena span:hover {
        transform: translateY(-2px);
    }

    .start-cue {
        margin-top: 18px;
        font-size: 0.96rem;
        font-weight: 650;
    }

    .results-arena {
        align-items: stretch;
        gap: 16px;
        margin-top: 18px;
    }

    .truth-card {
        padding: 14px !important;
        border-radius: 8px !important;
    }

    .model-card {
        position: relative;
        overflow: hidden;
        min-width: 260px;
        padding: 14px !important;
        border-radius: 8px !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .model-card:hover {
        transform: translateY(-4px) scale(1.01);
    }

    .model-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
    }

    .model-card h3 {
        margin-top: 8px !important;
    }

    .result-field {
        margin-top: 8px;
    }

    .result-field input,
    .result-field textarea,
    .truth-card input {
        border-radius: 8px !important;
        font-weight: 600;
    }

    #shuffle-images-button,
    #shuffle-images-button button {
        border-radius: 8px !important;
        color: #ffffff !important;
        border: 0 !important;
    }

    #shuffle-images-button:hover,
    #shuffle-images-button button:hover {
        filter: brightness(1.04);
        transform: translateY(-1px);
    }

    .gradio-container {
        background:
            radial-gradient(circle at 8% 0%, rgba(248, 250, 252, 0.14), transparent 22%),
            radial-gradient(circle at 86% 8%, rgba(100, 116, 139, 0.18), transparent 24%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.025) 25%, transparent 25%) 0 0 / 22px 22px,
            linear-gradient(180deg, #07080b 0%, #101319 34%, #e8ebef 34%, #f3f4f6 100%) !important;
        color: #111827;
        font-family: "Trebuchet MS", "Segoe UI", Arial, sans-serif;
    }

    .intro-banner {
        padding: 42px 44px !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        background:
            radial-gradient(circle at 78% 18%, rgba(255, 255, 255, 0.12), transparent 18%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.02)),
            repeating-linear-gradient(135deg, transparent 0 14px, rgba(255, 255, 255, 0.035) 14px 15px),
            linear-gradient(120deg, #050608 0%, #11151c 40%, #252b35 100%) !important;
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
    }

    .intro-banner::before {
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px) !important;
        opacity: 0.8;
    }

    .intro-banner::after {
        content: "";
        position: absolute;
        right: -18%;
        top: -70%;
        width: 56%;
        height: 190%;
        background: linear-gradient(90deg, transparent, rgba(203, 213, 225, 0.18), transparent);
        transform: rotate(18deg);
        pointer-events: none;
    }

    .intro-eyebrow {
        border-color: rgba(203, 213, 225, 0.42) !important;
        background: rgba(203, 213, 225, 0.08) !important;
        color: #e5e7eb !important;
        font-family: "Arial Black", "Segoe UI Black", sans-serif;
        letter-spacing: 0.12em;
    }

    .intro-banner h1 {
        color: #f8fafc !important;
        background: none !important;
        -webkit-text-fill-color: #f8fafc !important;
        font-family: Impact, Haettenschweiler, "Arial Black", "Segoe UI Black", sans-serif;
        font-size: clamp(3.8rem, 8vw, 7.2rem) !important;
        font-weight: 900;
        letter-spacing: 0.065em;
        line-height: 0.9 !important;
        text-transform: uppercase;
        text-shadow:
            1px 1px 0 #94a3b8,
            3px 4px 0 #1f2937,
            0 16px 34px rgba(0, 0, 0, 0.62);
    }

    .intro-banner h1::after {
        content: "";
        display: block;
        width: min(360px, 72vw);
        height: 4px;
        margin-top: 22px;
        background: linear-gradient(90deg, #ffffff, #94a3b8, rgba(203, 213, 225, 0));
        box-shadow: 0 0 22px rgba(203, 213, 225, 0.28);
    }

    .intro-banner p {
        color: #d7dde8 !important;
        font-size: 1.05rem !important;
        font-family: "Segoe UI", Arial, sans-serif;
        letter-spacing: 0.015em;
    }

    .intro-banner strong {
        color: #ffffff;
        font-weight: 760;
    }

    .model-arena span {
        border: 1px solid rgba(203, 213, 225, 0.24) !important;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.025)) !important;
        color: #eef2f7 !important;
        font-family: "Arial Black", "Segoe UI Black", sans-serif;
        letter-spacing: 0.035em;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    }

    .model-arena span:hover {
        border-color: rgba(203, 213, 225, 0.72) !important;
        box-shadow: 0 16px 28px rgba(0, 0, 0, 0.22) !important;
        color: #ffffff !important;
    }

    .start-cue {
        color: #cbd5e1 !important;
    }

    #shuffle-images-button,
    #shuffle-images-button button {
        background: linear-gradient(135deg, #111827 0%, #2d3748 56%, #cbd5e1 100%) !important;
        box-shadow: 0 14px 34px rgba(17, 24, 39, 0.28) !important;
        letter-spacing: 0.02em;
        font-weight: 760 !important;
    }

    .truth-card {
        border-color: rgba(203, 213, 225, 0.2) !important;
        background:
            linear-gradient(145deg, #f8fafc 0%, #d9dee7 100%) !important;
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.75) !important;
    }

    .model-card {
        border: 1px solid rgba(17, 24, 39, 0.1) !important;
        background:
            radial-gradient(circle at 18% 0%, rgba(255, 255, 255, 0.24), transparent 30%),
            radial-gradient(circle at 88% 16%, rgba(203, 213, 225, 0.16), transparent 24%),
            linear-gradient(145deg, rgba(96, 104, 118, 0.98) 0%, rgba(45, 51, 62, 0.98) 54%, rgba(28, 33, 42, 0.98) 100%) !important;
        box-shadow:
            0 16px 34px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.14),
            inset 0 -1px 0 rgba(0, 0, 0, 0.55) !important;
    }

    .model-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow:
            0 24px 54px rgba(0, 0, 0, 0.34),
            inset 0 1px 0 rgba(255, 255, 255, 0.18),
            0 0 0 1px rgba(226, 232, 240, 0.24),
            0 0 28px rgba(226, 232, 240, 0.1) !important;
    }

    .model-card::before {
        height: 4px;
        background: linear-gradient(90deg, transparent, #ffffff, #cbd5e1, #64748b, transparent) !important;
    }

    .model-card::after {
        content: none;
    }

    .model-card:hover::after {
        content: "Check performance";
        position: absolute;
        bottom: 8px;
        left: 12px;
        color: rgba(248, 250, 252, 0.72);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .model-card:has(.prediction-status.correct) {
        opacity: 1;
    }

    .model-card:has(.prediction-status.wrong) {
        opacity: 0.78;
        transition: opacity 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
    }

    .model-card:has(.prediction-status.correct):hover {
        box-shadow:
            0 24px 54px rgba(0, 0, 0, 0.34),
            inset 0 1px 0 rgba(255, 255, 255, 0.18),
            0 0 0 1px rgba(34, 197, 94, 0.24),
            0 0 30px rgba(34, 197, 94, 0.14) !important;
    }

    .model-card:has(.prediction-status.wrong):hover {
        box-shadow:
            0 24px 54px rgba(0, 0, 0, 0.34),
            inset 0 1px 0 rgba(255, 255, 255, 0.18),
            0 0 0 1px rgba(239, 68, 68, 0.2),
            0 0 26px rgba(239, 68, 68, 0.12) !important;
    }

    .prediction-status {
        position: absolute;
        top: 9px;
        right: 9px;
        display: grid;
        place-items: center;
        width: 16px;
        height: 16px;
        border-radius: 999px;
        background: #e5e7eb;
        color: transparent;
        font-family: "Arial Black", "Segoe UI Black", sans-serif;
        font-size: 0.78rem;
        line-height: 1;
        box-shadow:
            0 0 0 4px rgba(255, 255, 255, 0.08),
            0 0 16px rgba(229, 231, 235, 0.72);
        transition:
            width 0.16s ease,
            height 0.16s ease,
            background 0.16s ease,
            color 0.16s ease,
            transform 0.16s ease,
            box-shadow 0.16s ease;
        z-index: 2;
    }

    .status-holder {
        display: contents !important;
    }

    .winner-popup {
        position: sticky;
        top: 12px;
        z-index: 10;
        max-width: 620px;
        margin: 18px auto 8px;
        padding: 18px 22px;
        border: 1px solid rgba(255, 255, 255, 0.24);
        border-radius: 8px;
        background:
            radial-gradient(circle at 12% 0%, rgba(255, 255, 255, 0.22), transparent 24%),
            linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(75, 85, 99, 0.94));
        color: #ffffff;
        box-shadow:
            0 24px 56px rgba(0, 0, 0, 0.38),
            0 0 0 1px rgba(255, 255, 255, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.14);
        text-align: center;
        animation: winner-pop 0.58s cubic-bezier(0.18, 0.9, 0.25, 1.25) both;
    }

    .winner-popup::before,
    .winner-popup::after {
        content: "✦";
        position: absolute;
        top: 14px;
        width: auto;
        font-size: 1rem;
        color: #f8fafc;
        opacity: 0.62;
        pointer-events: none;
        text-shadow: 0 0 16px rgba(248, 250, 252, 0.65);
        animation: sparkle 1.35s ease-in-out infinite alternate;
    }

    .winner-popup::before {
        left: 18px;
    }

    .winner-popup::after {
        right: 18px;
        animation-delay: 0.25s;
    }

    .winner-popup.no-winner::before,
    .winner-popup.no-winner::after {
        content: "✕";
        color: #d1d5db;
        opacity: 0.38;
        animation: none;
    }

    .winner-emojis {
        margin-bottom: 4px;
        font-size: 1.55rem;
        animation: bounce 0.7s ease both;
    }

    .winner-kicker {
        color: #d1d5db;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .winner-title {
        margin-top: 2px;
        font-size: clamp(1.45rem, 3vw, 2.35rem);
        font-weight: 900;
        letter-spacing: 0.035em;
        text-transform: uppercase;
        text-shadow: 0 8px 20px rgba(0, 0, 0, 0.34);
    }

    .winner-detail {
        margin-top: 6px;
        color: #e5e7eb;
        font-size: 0.98rem;
        font-weight: 650;
    }

    .winner-popup.no-winner {
        background:
            linear-gradient(135deg, rgba(31, 41, 55, 0.96), rgba(75, 85, 99, 0.9));
        animation: winner-pop 0.42s ease both;
    }

    .winner-popup.no-winner .winner-title {
        font-size: clamp(1.2rem, 2.4vw, 1.75rem);
    }

    @keyframes winner-pop {
        0% {
            opacity: 0;
            transform: translateY(-18px) scale(0.84);
        }
        65% {
            opacity: 1;
            transform: translateY(2px) scale(1.04);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    @keyframes bounce {
        0% {
            transform: translateY(-8px) scale(0.75);
        }
        65% {
            transform: translateY(2px) scale(1.08);
        }
        100% {
            transform: translateY(0) scale(1);
        }
    }

    @keyframes sparkle {
        from {
            transform: translateY(0) scale(0.9);
            opacity: 0.42;
        }
        to {
            transform: translateY(3px) scale(1.16);
            opacity: 0.9;
        }
    }

    .model-card:hover .prediction-status {
        width: 34px;
        height: 34px;
        font-size: 1.2rem;
        color: #ffffff;
        transform: translate(5px, -5px);
    }

    .model-card:hover .prediction-status.correct {
        background: #22c55e;
        box-shadow:
            0 0 0 4px rgba(34, 197, 94, 0.16),
            0 0 20px rgba(34, 197, 94, 0.82);
        animation: pop 0.34s ease both;
    }

    .model-card:hover .prediction-status.wrong {
        background: #ef4444;
        box-shadow:
            0 0 0 4px rgba(239, 68, 68, 0.16),
            0 0 20px rgba(239, 68, 68, 0.82);
        animation: shake 0.34s ease both;
    }

    @keyframes pop {
        0% {
            transform: translate(5px, -5px) scale(0.58);
        }
        70% {
            transform: translate(5px, -5px) scale(1.18);
        }
        100% {
            transform: translate(5px, -5px) scale(1);
        }
    }

    @keyframes shake {
        0% {
            transform: translate(5px, -5px) translateX(0);
        }
        25% {
            transform: translate(5px, -5px) translateX(-4px);
        }
        50% {
            transform: translate(5px, -5px) translateX(4px);
        }
        75% {
            transform: translate(5px, -5px) translateX(-4px);
        }
        100% {
            transform: translate(5px, -5px) translateX(0);
        }
    }

    .model-card h3 {
        margin: 12px 0 16px !important;
        color: #f8fafc !important;
        font-family: "Trebuchet MS", "Segoe UI", Arial, sans-serif;
        font-size: 1.12rem !important;
        font-weight: 850 !important;
        letter-spacing: 0.02em;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.32);
    }

    .model-ann,
    .model-ann-optimized,
    .model-ann-hyperparameterized,
    .model-cnn,
    .model-resnet18-transfer-learning {
        --card-accent: linear-gradient(90deg, #111827, #cbd5e1, #f8fafc);
        background:
            linear-gradient(180deg, rgba(50, 56, 66, 0.98), rgba(21, 25, 32, 0.98)) !important;
    }

    .result-field label,
    .truth-card label {
        color: #d7dde8 !important;
        font-family: "Trebuchet MS", "Segoe UI", Arial, sans-serif;
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .result-field input,
    .result-field textarea,
    .truth-card input {
        border-color: rgba(203, 213, 225, 0.16) !important;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.055)),
            rgba(20, 24, 31, 0.72) !important;
        color: #f8fafc !important;
        font-family: "Trebuchet MS", "Segoe UI", Arial, sans-serif;
        font-size: 1rem !important;
        box-shadow:
            inset 0 2px 8px rgba(0, 0, 0, 0.32),
            0 1px 0 rgba(255, 255, 255, 0.07) !important;
        transition: border-color 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
    }

    .model-card:hover .result-field input,
    .model-card:hover .result-field textarea {
        border-color: rgba(248, 250, 252, 0.28) !important;
        box-shadow:
            inset 0 2px 8px rgba(0, 0, 0, 0.3),
            0 0 0 1px rgba(248, 250, 252, 0.08) !important;
    }

    .prediction-field input,
    .confidence-field input,
    .top3-field textarea,
    .time-field input {
        background:
            linear-gradient(180deg, rgba(148, 163, 184, 0.92), rgba(100, 116, 139, 0.88)) !important;
        color: #f8fafc !important;
    }

    .prediction-field input,
    .confidence-field input,
    .top3-field textarea,
    .time-field input,
    .prediction-field .wrap,
    .confidence-field .wrap,
    .top3-field .wrap,
    .time-field .wrap {
        border-color: rgba(203, 213, 225, 0.16) !important;
        background:
            linear-gradient(180deg, rgba(148, 163, 184, 0.92), rgba(100, 116, 139, 0.88)) !important;
        color: #f8fafc !important;
    }

    .prediction-field input {
        border-left: 0 !important;
    }

    .confidence-field input {
        border-left: 0 !important;
    }

    .top3-field textarea {
        border-left: 0 !important;
    }

    .time-field input {
        border-left: 0 !important;
    }

    .prediction-field input,
    .confidence-field input,
    .time-field input,
    .top3-field textarea,
    .top3-field textarea:disabled,
    .top3-field textarea[disabled] {
        opacity: 1 !important;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        background-color: #656c75 !important;
        background-image: linear-gradient(180deg, #757b84 0%, #555b63 100%) !important;
        box-shadow:
            inset 0 2px 8px rgba(0, 0, 0, 0.32),
            0 1px 0 rgba(255, 255, 255, 0.07) !important;
    }

    .truth-card label {
        color: #334155 !important;
    }

    .truth-card input {
        border-color: rgba(15, 23, 42, 0.14) !important;
        background: #ffffff !important;
        color: #111827 !important;
    }

    .selected-photo {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }

    .selected-photo img {
        filter: blur(1px);
        opacity: 0.82;
        transition: filter 0.2s ease, opacity 0.2s ease, transform 0.2s ease;
    }

    .selected-photo.active img {
        filter: none;
        opacity: 1;
    }

    #sample-gallery {
        padding: 12px !important;
        border: 1px solid rgba(203, 213, 225, 0.18) !important;
        border-radius: 8px !important;
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.035)),
            rgba(17, 24, 39, 0.72) !important;
    }

    #sample-gallery .grid-wrap,
    #sample-gallery .grid-container,
    #sample-gallery .thumbnails {
        gap: 12px !important;
    }

    #sample-gallery button,
    #sample-gallery .thumbnail-item,
    #sample-gallery [role="button"],
    #sample-gallery figure {
        overflow: hidden !important;
        border: 2px solid rgba(203, 213, 225, 0.16) !important;
        border-radius: 8px !important;
        background: #0f141b !important;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.22) !important;
        transition:
            transform 0.16s ease,
            border-color 0.16s ease,
            box-shadow 0.16s ease,
            background 0.16s ease;
    }

    #sample-gallery button:hover,
    #sample-gallery .thumbnail-item:hover,
    #sample-gallery [role="button"]:hover,
    #sample-gallery figure:hover {
        transform: translateY(-3px);
        border-color: rgba(248, 250, 252, 0.58) !important;
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.34), 0 0 0 1px rgba(248, 250, 252, 0.16) !important;
    }

    #sample-gallery img {
        transition: transform 0.18s ease, filter 0.18s ease;
    }

    #sample-gallery button:hover img,
    #sample-gallery .thumbnail-item:hover img,
    #sample-gallery [role="button"]:hover img,
    #sample-gallery figure:hover img {
        transform: scale(1.08);
        filter: contrast(1.08) brightness(1.05);
    }

    #sample-gallery .selected,
    #sample-gallery .thumbnail-item.selected,
    #sample-gallery [aria-selected="true"],
    #sample-gallery button.selected,
    #sample-gallery figure.selected {
        border-color: #ffffff !important;
        background: linear-gradient(180deg, #3f4651, #171b22) !important;
        box-shadow:
            0 0 0 3px rgba(255, 255, 255, 0.16),
            0 16px 34px rgba(0, 0, 0, 0.36) !important;
    }

    #sample-gallery button[aria-label*="Remove"],
    #sample-gallery button[aria-label*="Delete"],
    #sample-gallery button[title*="Remove"],
    #sample-gallery button[title*="Delete"],
    #sample-gallery .remove,
    #sample-gallery .delete {
        display: none !important;
        pointer-events: none !important;
    }

"""

INTERACTION_JS_TEMPLATE = r"""
const playAudio = (key, source, volume = 1) => {
    const storeKey = `__battle${key}Audio`;
    const audio = window[storeKey] || new Audio(source);
    window[storeKey] = audio;
    audio.pause();
    audio.volume = volume;
    audio.currentTime = 0;
    audio.play().catch(() => {});
};

const playSelectSound = () => playAudio("Select", __SELECT_SOUND_URL__, 1);

const winner = document.querySelector(".winner-popup");
const results = document.querySelector("#results-section");
if (winner) winner.scrollIntoView({ behavior: "smooth", block: "center" });
else if (results) results.scrollIntoView({ behavior: "smooth", block: "start" });

"""


def audio_data_url(audio_path):
    with open(audio_path, "rb") as audio_file:
        return (
            "data:audio/mpeg;base64,"
            + base64.b64encode(audio_file.read()).decode("ascii")
        )


SELECT_SOUND_URL = audio_data_url(SELECT_SOUND_PATH)
FEEDBACK_JS = INTERACTION_JS_TEMPLATE.replace(
    "__SELECT_SOUND_URL__",
    json.dumps(SELECT_SOUND_URL),
)

CSV_PATH = os.path.join(BASE_DIR, "dataset", "fmnist_small.csv")

ANN_MODEL_PATH = os.path.join(BASE_DIR, "pt_file", "ann_model.pt")
OPTIMIZED_ANN_MODEL_PATH = os.path.join(BASE_DIR, "pt_file", "ann_optimised_model (1).pt")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "pt_file", "cnn_model (6).pt")
RESNET18_MODEL_PATH = os.path.join(BASE_DIR, "pt_file", "resnet18_fmnist.pt")

HYPERPARAM_ANN_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "pt_file", "ann_hyperparameterimised_model.pt")
]

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class BasicANN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


class HyperparameterizedANN(nn.Module):
    def __init__(self, num_features, hidden_layer_sizes, dropout_rate=0.4):
        super().__init__()

        layers = []
        input_dim = num_features

        for hidden_size in hidden_layer_sizes:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                ]
            )
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class OptimizedANN(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


class CNNModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet18_model():
    model = models.resnet18(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
    )

    return model


def resolve_existing_path(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"None of these model files were found: {path_candidates}")


RESNET18_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def row_to_image_and_tensor(row):
    label = int(row.iloc[0])
    pixels = row.iloc[1:].to_numpy(dtype="uint8")

    image = Image.fromarray(pixels.reshape(28, 28), mode="L")

    flat_tensor = torch.tensor(pixels, dtype=torch.float32).div(255.0).unsqueeze(0)
    image_tensor = flat_tensor.view(1, 1, 28, 28)

    rgb_pixels = np.stack([pixels.reshape(28, 28)] * 3, axis=-1)
    rgb_image = Image.fromarray(rgb_pixels)

    resnet18_tensor = RESNET18_TRANSFORM(rgb_image).unsqueeze(0)

    return label, image, flat_tensor, image_tensor, resnet18_tensor


def load_class_samples():
    class_samples = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        row = dataset_df[dataset_df.iloc[:, 0] == class_index].sample(n=1).iloc[0]

        label, image, flat_tensor, image_tensor, resnet18_tensor = row_to_image_and_tensor(row)

        class_samples.append(
            {
                "label": label,
                "class_name": class_name,
                "image": image,
                "flat_tensor": flat_tensor,
                "image_tensor": image_tensor,
                "resnet18_tensor": resnet18_tensor,
                "gallery_caption": f"{class_index}: {class_name}",
            }
        )

    return class_samples


def load_model(model_class, model_path, model_input_features=784):
    state_dict = torch.load(model_path, map_location="cpu")

    model = model_class(model_input_features)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def build_hyperparameterized_model(num_features, state_dict):
    linear_layer_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("model.")
            and key.endswith(".weight")
            and state_dict[key].ndim == 2
        }
    )

    hidden_layer_sizes = [
        int(state_dict[f"model.{index}.weight"].shape[0])
        for index in linear_layer_indices[:-1]
    ]

    return HyperparameterizedANN(
        num_features,
        hidden_layer_sizes=hidden_layer_sizes,
    )


def load_hyperparameterized_model(model_path):
    state_dict = torch.load(model_path, map_location="cpu")

    model = build_hyperparameterized_model(784, state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_resnet18_model(model_path):
    state_dict = torch.load(model_path, map_location="cpu")

    model = build_resnet18_model()
    model.load_state_dict(state_dict)
    model.eval()

    return model


def run_inference(model, tensor):
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)

    predicted_index = int(top_indices[0, 0].item())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(top_probs[0, 0].item()) * 100

    top_3_predictions = [
        f"{CLASS_NAMES[int(index.item())]}: {float(prob.item()) * 100:.2f}%"
        for prob, index in zip(top_probs[0], top_indices[0])
    ]

    return predicted_class, confidence, top_3_predictions, inference_time_ms


def format_model_result(model, tensor, true_class_name):
    predicted_class, confidence, top_3_predictions, inference_time_ms = run_inference(
        model,
        tensor,
    )

    is_correct = predicted_class == true_class_name
    status_class = "correct" if is_correct else "wrong"
    status_symbol = "&#10003;" if is_correct else "&times;"
    status_label = "Correct prediction" if is_correct else "Wrong prediction"
    status_html = (
        f'<div class="prediction-status {status_class}" '
        f'aria-label="{status_label}" title="{status_label}">'
        f'<span>{status_symbol}</span></div>'
    )

    top_3_text = "\n".join(
        f"{rank}. {prediction}"
        for rank, prediction in enumerate(top_3_predictions, start=1)
    )

    return (
        (
            status_html,
            predicted_class,
            f"{confidence:.2f}%",
            top_3_text,
            f"{inference_time_ms:.2f} ms",
        ),
        predicted_class,
        confidence,
    )


def upscale_display_image(image, size=224):
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def build_winner_popup(winner, true_class_name):
    classes = "winner-popup no-winner"
    kicker = "No clear winner"
    title = "All models missed this round"
    detail = f"True label: {true_class_name}"
    emojis = ""

    if winner is not None:
        model_title, confidence = winner
        classes = "winner-popup"
        kicker = "Winner model"
        title = model_title
        detail = f"Matched the true label with {confidence:.2f}% confidence"
        emojis = '<div class="winner-emojis">&#127881; &#127942; &#10024;</div>'

    return (
        f'<div class="{classes}">'
        f"{emojis}"
        f'<div class="winner-kicker">{kicker}</div>'
        f'<div class="winner-title">{title}</div>'
        f'<div class="winner-detail">{detail}</div>'
        "</div>"
    )


def get_winner(winner_candidates):
    return (
        max(winner_candidates, key=lambda candidate: candidate[1])
        if winner_candidates
        else None
    )


def format_single_model_result(spec, sample):
    model_result, predicted_class, confidence = format_model_result(
        MODELS[spec["key"]]["model"],
        sample[spec["input_key"]],
        sample["class_name"],
    )
    winner_candidate = (
        (spec["title"], confidence)
        if predicted_class == sample["class_name"]
        else None
    )

    return model_result, winner_candidate


dataset_df = pd.read_csv(CSV_PATH)


MODEL_SPECS = [
    {
        "key": "ann",
        "title": "ANN",
        "model_class": BasicANN,
        "model_path": ANN_MODEL_PATH,
        "input_key": "flat_tensor",
    },
    {
        "key": "ann_optimized",
        "title": "Optimized ANN",
        "model_class": OptimizedANN,
        "model_path": OPTIMIZED_ANN_MODEL_PATH,
        "input_key": "flat_tensor",
    },
    {
        "key": "ann_hyperparameterized",
        "title": "Hyperparameterized ANN",
        "loader": load_hyperparameterized_model,
        "model_path": resolve_existing_path(HYPERPARAM_ANN_MODEL_CANDIDATES),
        "input_key": "flat_tensor",
    },
    {
        "key": "cnn",
        "title": "CNN",
        "model_class": CNNModel,
        "model_path": CNN_MODEL_PATH,
        "model_input_features": 1,
        "input_key": "image_tensor",
    },
    {
        "key": "resnet18_transfer_learning",
        "title": "ResNet18 Transfer Learning",
        "loader": load_resnet18_model,
        "model_path": RESNET18_MODEL_PATH,
        "input_key": "resnet18_tensor",
    },
]


MODELS = {
    spec["key"]: {
        **spec,
        "model": (
            spec["loader"](spec["model_path"])
            if "loader" in spec
            else load_model(
                spec["model_class"],
                spec["model_path"],
                spec.get("model_input_features", 784),
            )
        ),
    }
    for spec in MODEL_SPECS
}


def format_prediction_result(sample_index, current_samples):
    if current_samples is None:
        current_samples = load_class_samples()

    sample_index = 0 if sample_index is None else int(sample_index)
    sample = current_samples[sample_index]

    results = [
        upscale_display_image(sample["image"]),
        f'{sample["class_name"]} ({sample["label"]})',
    ]
    winner_candidates = []

    for spec in MODEL_SPECS:
        model_result, winner_candidate = format_single_model_result(spec, sample)
        results.extend(model_result)
        if winner_candidate is not None:
            winner_candidates.append(winner_candidate)

    results.insert(2, build_winner_popup(get_winner(winner_candidates), sample["class_name"]))

    return tuple(results)


def build_gallery_items(current_samples):
    return [
        (sample["image"], sample["gallery_caption"])
        for sample in current_samples
    ]


def initialize_session():
    current_samples = load_class_samples()
    return current_samples, gr.Gallery(value=build_gallery_items(current_samples), selected_index=0), *format_prediction_result(
        0, current_samples
    )


def on_gallery_select(current_samples, evt: gr.SelectData):
    return format_prediction_result(evt.index, current_samples)


with gr.Blocks() as demo:
    result_feedback_js = f"""
    (samples) => {{
        {FEEDBACK_JS}
        document.querySelector(".selected-photo")?.classList.add("active");
        if (window.__battleSuppressNextSelectSound) {{
            window.__battleSuppressNextSelectSound = false;
        }} else {{
            playSelectSound();
        }}
        return samples;
    }}
    """

    shuffle_feedback_js = f"""
    () => {{
        window.__battleSuppressNextSelectSound = true;
        window.setTimeout(() => {{
            window.__battleSuppressNextSelectSound = false;
        }}, 1800);
        {FEEDBACK_JS}
        document.querySelector(".selected-photo")?.classList.remove("active");
    }}
    """

    gr.Markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%); padding: 42px 44px; border-radius: 8px; margin-bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.16); box-shadow: 0 28px 70px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.12);">
            <div style="display: inline-flex; align-items: center; gap: 8px; margin-bottom: 12px; padding: 6px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; border: 1px solid rgba(203, 213, 225, 0.42); background: rgba(203, 213, 225, 0.08); color: #e5e7eb; font-family: 'Arial Black', 'Segoe UI Black', sans-serif; letter-spacing: 0.12em;">Live prediction arena</div>
            <h1 style="margin: 0 0 14px; width: fit-content; color: #f8fafc; background: none; -webkit-text-fill-color: #f8fafc; font-family: Impact, Haettenschweiler, 'Arial Black', 'Segoe UI Black', sans-serif; font-size: clamp(3.8rem, 8vw, 7.2rem); font-weight: 900; letter-spacing: 0.065em; line-height: 0.9; text-transform: uppercase; text-shadow: 1px 1px 0 #94a3b8, 3px 4px 0 #1f2937, 0 16px 34px rgba(0, 0, 0, 0.62);">Battle of the Models</h1>
            <p style="max-width: 760px; margin: 0 0 8px; line-height: 1.55; color: #d7dde8; font-size: 1.05rem; font-family: 'Segoe UI', Arial, sans-serif; letter-spacing: 0.015em;"><strong style="color: #ffffff; font-weight: 760;">Not all models think the same.</strong> Compare predictions across various models in real time.</p>
            <p style="max-width: 760px; margin: 0 0 8px; line-height: 1.55; color: #d7dde8; font-size: 1.05rem; font-family: 'Segoe UI', Arial, sans-serif; letter-spacing: 0.015em;">Pick an image and see where models agree — and where they fail.</p>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px;">
                <span style="padding: 9px 12px; border-radius: 8px; font-size: 0.92rem; font-weight: 650; transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, color 0.16s ease; border: 1px solid rgba(203, 213, 225, 0.24); background: linear-gradient(180deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.025)); color: #eef2f7; font-family: 'Arial Black', 'Segoe UI Black', sans-serif; letter-spacing: 0.035em; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);">ANN</span>
                <span style="padding: 9px 12px; border-radius: 8px; font-size: 0.92rem; font-weight: 650; transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, color 0.16s ease; border: 1px solid rgba(203, 213, 225, 0.24); background: linear-gradient(180deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.025)); color: #eef2f7; font-family: 'Arial Black', 'Segoe UI Black', sans-serif; letter-spacing: 0.035em; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);">Optimized ANN</span>
                <span style="padding: 9px 12px; border-radius: 8px; font-size: 0.92rem; font-weight: 650; transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, color 0.16s ease; border: 1px solid rgba(203, 213, 225, 0.24); background: linear-gradient(180deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.025)); color: #eef2f7; font-family: 'Arial Black', 'Segoe UI Black', sans-serif; letter-spacing: 0.035em; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);">Hyperparameterized ANN</span>
                <span style="padding: 9px 12px; border-radius: 8px; font-size: 0.92rem; font-weight: 650; transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, color 0.16s ease; border: 1px solid rgba(203, 213, 225, 0.24); background: linear-gradient(180deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.025)); color: #eef2f7; font-family: 'Arial Black', 'Segoe UI Black', sans-serif; letter-spacing: 0.035em; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);">CNN</span>
                <span style="padding: 9px 12px; border-radius: 8px; font-size: 0.92rem; font-weight: 650; transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, color 0.16s ease; border: 1px solid rgba(203, 213, 225, 0.24); background: linear-gradient(180deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.025)); color: #eef2f7; font-family: 'Arial Black', 'Segoe UI Black', sans-serif; letter-spacing: 0.035em; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);">ResNet18</span>
            </div>
            <div style="margin-top: 18px; font-size: 0.96rem; font-weight: 650; color: #cbd5e1;">Pick an image below and watch the scoreboard change.</div>
        </div>
        """
    )

    session_samples = gr.State()
    gallery = gr.Gallery(
        value=[],
        label="Pick an image",
        columns=5,
        rows=2,
        allow_preview=False,
        interactive=True,
        height="auto",
        elem_id="sample-gallery",
    )
    refresh_button = gr.Button("Shuffle images", elem_id="shuffle-images-button")
    winner_popup = gr.HTML()

    with gr.Row(elem_classes=["results-arena"], elem_id="results-section"):
        selected_image = gr.Image(
            label="Selected image",
            interactive=False,
            height=260,
            elem_classes=["selected-photo"],
        )
        with gr.Column(elem_classes=["truth-card"]):
            true_label_text = gr.Textbox(
                label="True class",
                interactive=False,
                elem_classes=["result-field"],
            )
        model_result_components = []
        for spec in MODEL_SPECS:
            card_class = f"model-{spec['key'].replace('_', '-')}"
            with gr.Column(elem_classes=["model-card", card_class]):
                status_indicator = gr.HTML(elem_classes=["status-holder"])
                gr.Markdown(f"### {spec['title']} Scorecard")
                predicted_class_text = gr.Textbox(
                    label="Predicted class",
                    interactive=False,
                    elem_classes=["result-field", "prediction-field"],
                )
                confidence_text = gr.Textbox(
                    label="Confidence",
                    interactive=False,
                    elem_classes=["result-field", "confidence-field"],
                )
                top_3_text = gr.Textbox(
                    label="Top-3 predictions",
                    lines=3,
                    interactive=False,
                    elem_classes=["result-field", "top3-field"],
                )
                inference_time_text = gr.Textbox(
                    label="Inference time",
                    interactive=False,
                    elem_classes=["result-field", "time-field"],
                )
                model_result_components.extend(
                    [
                        status_indicator,
                        predicted_class_text,
                        confidence_text,
                        top_3_text,
                        inference_time_text,
                    ]
                )

    load_outputs = [
        session_samples,
        gallery,
        selected_image,
        true_label_text,
        winner_popup,
        *model_result_components,
    ]
    prediction_outputs = [
        selected_image,
        true_label_text,
        winner_popup,
        *model_result_components,
    ]

    demo.load(
        fn=initialize_session,
        outputs=load_outputs,
    )

    gallery.select(
        fn=on_gallery_select,
        inputs=[session_samples],
        outputs=prediction_outputs,
        js=result_feedback_js,
    )

    refresh_button.click(
        fn=initialize_session,
        outputs=load_outputs,
        js=shuffle_feedback_js,
    )

    gr.Markdown(
        "This app loads every model listed in `MODEL_SPECS`, runs all inferences on the selected Fashion-MNIST sample, and displays the results side by side. To add another model later, add one more entry to `MODEL_SPECS`."
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(show_error=True, css=APP_CSS)

