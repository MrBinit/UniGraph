import { useEffect, useState } from "react";

import { BrandLogo } from "./Brand";
import {
  CloseIcon,
  MoonIcon,
  MoreHorizontalIcon,
  PinIcon,
  PlusIcon,
  SearchIcon,
  SettingsIcon,
  StarIcon,
  SunIcon,
  TrashIcon,
} from "./Icons";
import type { AuthSession, ConversationItem } from "../types";

type ConversationDateFilter = "all" | "7d" | "30d";

interface SidebarProps {
  auth: AuthSession;
  conversations: ConversationItem[];
  recentConversations: ConversationItem[];
  activeConversationId: string;
  searchQuery: string;
  onSearchChange: (value: string) => void;
  dateFilter: ConversationDateFilter;
  onDateFilterChange: (value: ConversationDateFilter) => void;
  onNewChat: () => void;
  onDeleteConversation: (conversation: ConversationItem) => Promise<void>;
  deletingConversationId: string | null;
  isPinned: (conversationId: string) => boolean;
  isStarred: (conversationId: string) => boolean;
  onTogglePin: (conversationId: string) => void;
  onToggleStar: (conversationId: string) => void;
  onRenameConversation: (conversation: ConversationItem) => void;
  onSelectConversation: (conversation: ConversationItem) => void;
  onToggleTheme: () => void;
  darkMode: boolean;
  onLogout: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

function initials(name: string): string {
  const clean = name.trim();
  if (!clean) return "AI";
  return clean.slice(0, 2).toUpperCase();
}

function ConversationButton({
  item,
  active,
  menuOpen,
  deleting,
  pinned,
  starred,
  onToggleMenu,
  onDelete,
  onTogglePin,
  onToggleStar,
  onRename,
  onClick,
}: {
  item: ConversationItem;
  active: boolean;
  menuOpen: boolean;
  deleting: boolean;
  pinned: boolean;
  starred: boolean;
  onToggleMenu: () => void;
  onDelete: () => Promise<void>;
  onTogglePin: () => void;
  onToggleStar: () => void;
  onRename: () => void;
  onClick: () => void;
}) {
  return (
    <div className="group relative">
      <button
        type="button"
        onClick={onClick}
        className={`w-full rounded-xl px-3 py-2 pr-10 text-left text-sm transition ${
          active
            ? "bg-gradient-to-r from-brand-blue to-brand-red text-white shadow"
            : "bg-white/70 text-slate-700 hover:bg-blue-50 dark:bg-slate-800/60 dark:text-slate-200 dark:hover:bg-slate-700"
        }`}
      >
        <p className="truncate font-medium">
          {starred ? "★ " : ""}
          {pinned ? "📌 " : ""}
          {item.title}
        </p>
        <p className={`truncate text-xs ${active ? "text-blue-100" : "text-slate-500 dark:text-slate-400"}`}>
          {new Date(item.createdAt).toLocaleDateString()}
        </p>
      </button>

      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onToggleMenu();
        }}
        className={`absolute right-2 top-2 rounded-md p-1.5 transition ${
          active
            ? "text-blue-100 hover:bg-white/20"
            : "text-slate-500 hover:bg-blue-100 hover:text-brand-blue dark:text-slate-300 dark:hover:bg-slate-700"
        } ${menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}
        aria-label={`Conversation options for ${item.title}`}
      >
        <MoreHorizontalIcon className="h-4 w-4" />
      </button>

      {menuOpen ? (
        <div className="absolute right-2 top-11 z-20 w-44 rounded-xl border border-blue-100 bg-white p-1.5 shadow-xl dark:border-slate-700 dark:bg-slate-900">
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onTogglePin();
            }}
            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-slate-700 transition hover:bg-blue-50 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            <PinIcon className="h-4 w-4" />
            {pinned ? "Unpin" : "Pin"}
          </button>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onToggleStar();
            }}
            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-slate-700 transition hover:bg-blue-50 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            <StarIcon className="h-4 w-4" />
            {starred ? "Unstar" : "Star"}
          </button>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onRename();
            }}
            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-slate-700 transition hover:bg-blue-50 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            <SettingsIcon className="h-4 w-4" />
            Rename
          </button>
          <button
            type="button"
            onClick={async (event) => {
              event.stopPropagation();
              await onDelete();
            }}
            disabled={deleting}
            className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-rose-600 transition hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-60 dark:text-rose-300 dark:hover:bg-rose-950/40"
          >
            <TrashIcon className="h-4 w-4" />
            {deleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      ) : null}
    </div>
  );
}

export function Sidebar({
  auth,
  conversations,
  recentConversations,
  activeConversationId,
  searchQuery,
  onSearchChange,
  dateFilter,
  onDateFilterChange,
  onNewChat,
  onDeleteConversation,
  deletingConversationId,
  isPinned,
  isStarred,
  onTogglePin,
  onToggleStar,
  onRenameConversation,
  onSelectConversation,
  onToggleTheme,
  darkMode,
  onLogout,
  mobileOpen,
  onCloseMobile,
}: SidebarProps) {
  const [menuConversationId, setMenuConversationId] = useState<string | null>(null);

  useEffect(() => {
    const handleDocumentClick = () => {
      setMenuConversationId(null);
    };
    document.addEventListener("click", handleDocumentClick);
    return () => {
      document.removeEventListener("click", handleDocumentClick);
    };
  }, []);

  return (
    <>
      <div
        className={`fixed inset-0 z-30 bg-slate-900/30 transition md:hidden ${
          mobileOpen ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        onClick={onCloseMobile}
      />
      <aside
        className={`fixed left-0 top-0 z-40 flex h-screen w-[220px] shrink-0 flex-col border-r border-blue-100 bg-[#f7fbff] p-3 shadow-xl shadow-blue-200/40 transition-transform dark:border-slate-800 dark:bg-slate-950 md:z-10 md:translate-x-0 md:shadow-none ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div>
          <div className="mb-4 flex items-center justify-between">
            <BrandLogo compact className="h-7 max-w-[124px]" />
            <button
              type="button"
              className="rounded-lg p-1.5 text-slate-500 hover:bg-blue-50 hover:text-brand-blue dark:text-slate-300 dark:hover:bg-slate-800 md:hidden"
              onClick={onCloseMobile}
              aria-label="Close sidebar"
            >
              <CloseIcon className="h-5 w-5" />
            </button>
          </div>

          <div className="mb-2 flex items-center gap-2">
            <button
              type="button"
              onClick={onNewChat}
              className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-brand-blue to-brand-red px-3 py-2 text-sm font-medium text-white shadow-md shadow-blue-500/30"
            >
              <PlusIcon className="h-4 w-4" />
              New Chat
            </button>
            <button
              type="button"
              className="rounded-xl border border-blue-200 bg-white p-2 text-slate-600 hover:text-brand-blue dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
              aria-label="Search"
            >
              <SearchIcon className="h-4 w-4" />
            </button>
          </div>

          <input
            value={searchQuery}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="Search conversations"
            className="mb-3 w-full rounded-xl border border-blue-100 bg-white px-3 py-2 text-sm outline-none focus:border-blue-300 focus:ring-2 focus:ring-blue-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
          />
          <select
            value={dateFilter}
            onChange={(event) => onDateFilterChange(event.target.value as ConversationDateFilter)}
            className="mb-2 w-full rounded-xl border border-blue-100 bg-white px-3 py-2 text-xs text-slate-600 outline-none focus:border-blue-300 focus:ring-2 focus:ring-blue-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
          >
            <option value="all">All dates</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
          </select>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto pr-1">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Your conversations
          </p>
          <div className="space-y-2">
            {conversations.length ? (
              conversations.map((item) => (
                <ConversationButton
                  key={`primary-${item.conversationId}`}
                  item={item}
                  active={item.conversationId === activeConversationId}
                  menuOpen={menuConversationId === `primary-${item.conversationId}`}
                  deleting={deletingConversationId === item.conversationId}
                  pinned={isPinned(item.conversationId)}
                  starred={isStarred(item.conversationId)}
                  onToggleMenu={() => {
                    setMenuConversationId((prev) =>
                      prev === `primary-${item.conversationId}` ? null : `primary-${item.conversationId}`
                    );
                  }}
                  onTogglePin={() => {
                    onTogglePin(item.conversationId);
                    setMenuConversationId(null);
                  }}
                  onToggleStar={() => {
                    onToggleStar(item.conversationId);
                    setMenuConversationId(null);
                  }}
                  onRename={() => {
                    onRenameConversation(item);
                    setMenuConversationId(null);
                  }}
                  onDelete={async () => {
                    await onDeleteConversation(item);
                    setMenuConversationId(null);
                  }}
                  onClick={() => {
                    onSelectConversation(item);
                    setMenuConversationId(null);
                    onCloseMobile();
                  }}
                />
              ))
            ) : (
              <p className="rounded-xl border border-dashed border-blue-200 p-3 text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400">
                No conversations yet.
              </p>
            )}
          </div>

          <p className="mb-2 mt-6 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Last 7 Days
          </p>
          <div className="space-y-2">
            {recentConversations.length ? (
              recentConversations.map((item) => (
                <ConversationButton
                  key={`recent-${item.conversationId}`}
                  item={item}
                  active={item.conversationId === activeConversationId}
                  menuOpen={menuConversationId === `recent-${item.conversationId}`}
                  deleting={deletingConversationId === item.conversationId}
                  pinned={isPinned(item.conversationId)}
                  starred={isStarred(item.conversationId)}
                  onToggleMenu={() => {
                    setMenuConversationId((prev) =>
                      prev === `recent-${item.conversationId}` ? null : `recent-${item.conversationId}`
                    );
                  }}
                  onTogglePin={() => {
                    onTogglePin(item.conversationId);
                    setMenuConversationId(null);
                  }}
                  onToggleStar={() => {
                    onToggleStar(item.conversationId);
                    setMenuConversationId(null);
                  }}
                  onRename={() => {
                    onRenameConversation(item);
                    setMenuConversationId(null);
                  }}
                  onDelete={async () => {
                    await onDeleteConversation(item);
                    setMenuConversationId(null);
                  }}
                  onClick={() => {
                    onSelectConversation(item);
                    setMenuConversationId(null);
                    onCloseMobile();
                  }}
                />
              ))
            ) : (
              <p className="rounded-xl border border-dashed border-blue-200 p-3 text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400">
                No recent conversations.
              </p>
            )}
          </div>
        </div>

        <div className="mt-3 space-y-2 border-t border-blue-100 pt-3 dark:border-slate-800">
          <button
            type="button"
            className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm text-slate-700 hover:bg-blue-50 dark:text-slate-200 dark:hover:bg-slate-800"
            onClick={onToggleTheme}
          >
            {darkMode ? <SunIcon className="h-4 w-4" /> : <MoonIcon className="h-4 w-4" />}
            {darkMode ? "Light mode" : "Dark mode"}
          </button>

          <button
            type="button"
            className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm text-slate-700 hover:bg-blue-50 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            <SettingsIcon className="h-4 w-4" />
            Settings
          </button>
          <button
            type="button"
            onClick={onLogout}
            className="flex w-full items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
          >
            Logout
          </button>

          <div className="mt-2 flex items-center gap-3 rounded-2xl bg-white p-3 shadow dark:bg-slate-900">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-brand-blue to-brand-red text-sm font-semibold text-white">
              {initials(auth.username)}
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">{auth.username}</p>
              <p className="truncate text-xs text-slate-500 dark:text-slate-400">{auth.userId}</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
