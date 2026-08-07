"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#050816] disabled:pointer-events-none disabled:opacity-50 cursor-pointer select-none",
  {
    variants: {
      variant: {
        default:
          "bg-blue-600 text-white shadow-[0_0_0_1px_rgba(37,99,235,0.4),0_4px_16px_rgba(37,99,235,0.3)] hover:bg-blue-500 hover:shadow-[0_0_0_1px_rgba(37,99,235,0.6),0_6px_24px_rgba(37,99,235,0.45)] hover:-translate-y-0.5 active:translate-y-0",
        secondary:
          "bg-white/[0.04] text-slate-300 border border-white/[0.08] hover:bg-white/[0.08] hover:border-white/[0.15] hover:text-white hover:-translate-y-0.5",
        ghost:
          "text-slate-400 hover:bg-white/[0.05] hover:text-white",
        danger:
          "bg-red-600/90 text-white shadow-[0_0_0_1px_rgba(239,68,68,0.4)] hover:bg-red-500 hover:shadow-[0_0_0_1px_rgba(239,68,68,0.6),0_4px_16px_rgba(239,68,68,0.3)]",
        success:
          "bg-emerald-600/90 text-white shadow-[0_0_0_1px_rgba(34,197,94,0.4)] hover:bg-emerald-500",
        outline:
          "border border-white/[0.1] bg-transparent text-slate-300 hover:bg-white/[0.04] hover:border-white/[0.2] hover:text-white",
      },
      size: {
        sm: "h-8 px-3 text-xs rounded-lg",
        default: "h-10 px-5 py-2",
        lg: "h-12 px-8 text-base rounded-2xl",
        xl: "h-14 px-10 text-base rounded-2xl",
        icon: "h-10 w-10 rounded-lg",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
