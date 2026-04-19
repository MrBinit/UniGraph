import { SendIcon } from "./Icons";
import type { ChatExecutionMode } from "../types";

const MODE_OPTIONS: Array<{ value: ChatExecutionMode; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "standard", label: "Standard" },
  { value: "deep", label: "Deep" },
];

interface ChatInputProps {
  value: string;
  disabled: boolean;
  mode: ChatExecutionMode;
  suggestionChips: string[];
  autoScrollPaused: boolean;
  onChange: (value: string) => void;
  onModeChange: (mode: ChatExecutionMode) => void;
  onSuggestionClick: (value: string) => Promise<void>;
  onToggleAutoScroll: () => void;
  onSubmit: () => Promise<void>;
}

export function ChatInput({
  value,
  disabled,
  mode,
  suggestionChips,
  autoScrollPaused,
  onChange,
  onModeChange,
  onSuggestionClick,
  onToggleAutoScroll,
  onSubmit,
}: ChatInputProps) {
  return (
    <div className="mx-auto w-full max-w-4xl px-4 pb-2">
      {suggestionChips.length ? (
        <div className="mb-1 flex gap-1 overflow-x-auto whitespace-nowrap pb-0.5 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {suggestionChips.map((chip) => (
            <button
              key={chip}
              type="button"
              disabled={disabled}
              onClick={async () => {
                await onSuggestionClick(chip);
              }}
              className="rounded-full border border-blue-200 bg-white px-2.5 py-0.5 text-[11px] text-slate-600 hover:border-blue-300 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              {chip}
            </button>
          ))}
        </div>
      ) : null}
      <form
        className="flex items-end gap-2 rounded-xl border border-blue-200 bg-white/95 p-2 shadow-soft backdrop-blur dark:border-slate-700 dark:bg-slate-900"
        onSubmit={async (event) => {
          event.preventDefault();
          await onSubmit();
        }}
      >
        <div className="w-full">
          <div className="mb-1 flex items-center gap-1">
            {MODE_OPTIONS.map((option) => {
              const selected = mode === option.value;
              return (
                <button
                  key={option.value}
                  type="button"
                  disabled={disabled}
                  onClick={() => onModeChange(option.value)}
                  className={[
                    "rounded-md border px-2 py-0.5 text-[11px] font-medium transition",
                    selected
                      ? option.value === "deep"
                        ? "border-brand-red bg-rose-50 text-brand-red dark:border-rose-400 dark:bg-rose-950/30 dark:text-rose-200"
                        : "border-brand-blue bg-blue-50 text-brand-blue dark:border-blue-400 dark:bg-blue-950/40 dark:text-blue-200"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800",
                    disabled ? "cursor-not-allowed opacity-60" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  aria-label={`Use ${option.label.toLowerCase()} mode`}
                >
                  {option.label}
                </button>
              );
            })}
          </div>
          <textarea
            value={value}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={async (event) => {
              if (event.key !== "Enter" || event.shiftKey) {
                return;
              }
              event.preventDefault();
              await onSubmit();
            }}
            placeholder="What's in your mind?"
            className="max-h-24 min-h-8 w-full resize-y border-0 bg-transparent p-0.5 text-sm text-slate-800 outline-none placeholder:text-slate-400 disabled:opacity-60 dark:text-slate-100 dark:placeholder:text-slate-500"
          />
        </div>

        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-r from-brand-blue to-brand-red text-white shadow-lg shadow-blue-500/30 transition hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-45"
          aria-label="Send"
        >
          <SendIcon className="h-4 w-4" />
        </button>
      </form>
      <div className="mt-0.5 flex items-center justify-end px-1 text-[11px] text-slate-500 dark:text-slate-400">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onToggleAutoScroll}
            className="rounded border border-slate-200 px-1.5 py-0.5 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            {autoScrollPaused ? "Resume scroll" : "Pause scroll"}
          </button>
          {value.length ? (
            <span className={value.length > 5000 ? "font-semibold text-rose-500" : ""}>{value.length}</span>
          ) : null}
        </div>
      </div>
    </div>
  );
}
